from collections import OrderedDict
from os import cpu_count

import numpy as np
from mmcv.utils.progressbar import track_iter_progress, track_parallel_progress

from mmdet.core.evaluation.mean_ap import average_precision

import torch
from terminaltables import AsciiTable
from mmcv.utils import print_log

from .builder import (build_eval_matcher, build_eval_affinity_calculator,
                      build_eval_breakdown, build_eval_tp_metric)


class FlexibleStatisticsEval(object):

    def __init__(self, classes, match_thrs, breakdown, affinity_calculator,
                 matcher, tp_metrics, nproc):
        self.classes = classes
        self.breakdown = [
            build_eval_breakdown({'type': 'NoBreakdown'}, classes=classes)]
        self.breakdown += [
            build_eval_breakdown(bkd, classes=classes)
            for bkd in breakdown
        ]
        self.affinity_calculator = build_eval_affinity_calculator(
            affinity_calculator)
        self.matcher = build_eval_matcher(
            matcher,
            match_thrs=match_thrs,
            affinity_cost_negate=self.affinity_calculator.LARGER_CLOSER)
        self.tp_metric = [build_eval_tp_metric(m) for m in tp_metrics]
        self.nproc = nproc

    def statistics_single(self, input):
        """Check if detected bboxes are true positive or false positive."""
        tp_score_info = []
        det, gt = input
        num_cls = len(self.classes)
        num_match_thrs = len(self.matcher.match_thrs)

        attrs = [k for k in det if k in gt and k not in ['labels']]

        for cls in range(num_cls):
            # prepare detections
            cls_name = self.classes[cls] if self.classes is not None else cls
            cls_det_msk = det['labels'] == cls
            cls_det = {k: v[cls_det_msk] for k, v in det.items()}
            sort_ind = cls_det['scores'].argsort()[::-1]
            cls_det = {k: v[sort_ind] for k, v in cls_det.items()}
            cls_num_dets = cls_det['bboxes'].shape[0]

            # prepare ground-truths
            cls_gt_msk = gt['labels'] == cls
            cls_gt = {k: v[cls_gt_msk] for k, v in gt.items()}

            # prepare breakdown masks
            cls_det_bkd = []
            cls_gt_bkd = []
            cls_bkd_names = []
            for fun in self.breakdown:
                cls_det_bkd.append(fun.breakdown(cls_det, cls))
                cls_gt_bkd.append(fun.breakdown(cls_gt, cls))
                cls_bkd_names += fun.breakdown_names(cls)
            cls_det_bkd = np.concatenate(cls_det_bkd, axis=0)
            cls_gt_bkd = np.concatenate(cls_gt_bkd, axis=0)
            num_bkd = cls_gt_bkd.shape[0]

            # calculate num gt (not considering ignored gt boxes)
            cls_gt_count = []
            for bkd_idx in range(num_bkd):
                cls_gt_count.append(np.count_nonzero(cls_gt_bkd[bkd_idx]))

            # handling empty det or empty gt
            if cls_gt['bboxes'].shape[0] == 0 or cls_num_dets == 0:
                for bkd_idx in range(num_bkd):
                    tp_pairs = {k: [] for k in attrs}
                    for _ in range(num_match_thrs):
                        for k in attrs:
                            tp_pairs[k].append(
                                np.empty((0, 2) + cls_det[k].shape[1:],
                                         dtype=cls_det[k].dtype))

                    cls_tp = np.zeros((num_match_thrs, cls_num_dets),
                                      dtype=np.bool)

                    tp_score_info.append(
                        (cls_name, cls_bkd_names[bkd_idx],
                         cls_gt_count[bkd_idx], cls_det['scores'], cls_tp,
                         cls_det_bkd[bkd_idx:bkd_idx + 1].repeat(
                             num_match_thrs, axis=0), tp_pairs))
            else:
                affinity = self.affinity_calculator(cls_det, cls_gt)

                for bkd_idx in range(num_bkd):
                    cls_gt_bkd_msk = cls_gt_bkd[bkd_idx]
                    bkd_cls_gt = cls_gt.copy()
                    bkd_cls_gt['ignore'] = (~cls_gt_bkd_msk)

                    matched_gt_idx = self.matcher(affinity, bkd_cls_gt)

                    _msk_fp = (
                        cls_det_bkd[bkd_idx:bkd_idx + 1] &
                        (matched_gt_idx == -1))
                    _msk_tp = ((cls_gt_bkd_msk[matched_gt_idx]) &
                               (matched_gt_idx > -1))
                    _msk_fptp = (_msk_fp | _msk_tp)

                    tp_pairs = {k: [] for k in attrs}

                    for _m, _gt_idx in zip(_msk_tp, matched_gt_idx):
                        for k in attrs:
                            k_tp_pairs = np.stack(
                                [cls_det[k][_m], cls_gt[k][_gt_idx[_m]]],
                                axis=1)
                            tp_pairs[k].append(k_tp_pairs)

                    tp_score_info.append((cls_name, cls_bkd_names[bkd_idx],
                                          cls_gt_count[bkd_idx],
                                          cls_det['scores'], _msk_tp, _msk_fptp,
                                          tp_pairs))

        return tp_score_info

    def statistics_accumulate(self, input):
        cls, bkd, num_gt, score, tp, bkd_msk, tp_pairs = input
        eval_result_list = []
        rank = score.argsort()[::-1]
        _tp_pairs_rank = tp.cumsum(axis=1)[:, rank]
        tp = tp[:, rank]
        bkd_msk = bkd_msk[:, rank]
        _tp_pairs_rank = [_r[_tp] - 1 for _r, _tp in zip(_tp_pairs_rank, tp)]

        for match_thr_idx, match_thr in enumerate(self.matcher.match_thrs):
            key = OrderedDict(Class=cls, Breakdown=bkd, Thres=match_thr)
            thr_tp_pairs = {k: tp_pairs[k][match_thr_idx][
                _tp_pairs_rank[match_thr_idx]] for k in tp_pairs}
            thr_tp_metric = {tp_metric.name: tp_metric(thr_tp_pairs, key=key)
                             for tp_metric in self.tp_metric}
            tpcumsum = tp[match_thr_idx, bkd_msk[match_thr_idx]].cumsum()
            num_det = len(tpcumsum)
            recall = tpcumsum / max(num_gt, 1e-7)
            precision = tpcumsum / np.arange(1, num_det + 1)
            m_ap = average_precision(recall, precision)
            max_recall = recall.max() if len(recall) > 0 else 0
            value = OrderedDict(
                Dets=num_det, GTs=num_gt, Recall=max_recall, mAP=m_ap)
            value.update(thr_tp_metric)
            eval_result_list.append((key, value))
        return eval_result_list

    def statistics_eval(self, det_results, annotations):
        if self.nproc == 0:
            tp_score_infos = [
                self.statistics_single(d)
                for d in zip(track_iter_progress(det_results), annotations)
            ]
        else:
            tp_score_infos = track_parallel_progress(
                func=self.statistics_single,
                tasks=[*zip(det_results, annotations)],
                nproc=self.nproc,
                chunksize=16)

        tp_score_infos_all = []
        for cls_bkd_item in zip(*tp_score_infos):
            (cls, bkd, num_gt, score, tp, bkd_msk,
             _tp_pairs) = tuple(zip(*cls_bkd_item))
            assert len(set(cls)) == 1
            assert len(set(bkd)) == 1
            num_gt = sum(num_gt)
            score = np.concatenate(score, axis=0)
            tp = np.concatenate(tp, axis=1)
            bkd_msk = np.concatenate(bkd_msk, axis=1)

            tp_pairs = dict()
            for k in _tp_pairs[0]:
                tp_pairs[k] = [_pairs[k] for _pairs in _tp_pairs]
                tp_pairs[k] = [np.concatenate(_pairs, axis=0)
                               for _pairs in zip(*tp_pairs[k])]

            tp_score_infos_all.append(
                (cls[0], bkd[0], num_gt, score, tp, bkd_msk, tp_pairs))

        if self.nproc == 0:
            eval_result_list = [
                self.statistics_accumulate(d)
                for d in track_iter_progress(tp_score_infos_all)
            ]
        else:
            eval_result_list = track_parallel_progress(
                func=self.statistics_accumulate,
                tasks=tp_score_infos_all,
                nproc=self.nproc,
                chunksize=16)

        eval_result_list = sum(eval_result_list, [])
        return eval_result_list

    def report(self, eval_result_list, group_by):
        report_dict = OrderedDict()
        for name, cond in group_by:
            cond_met_map = []
            for k, v in eval_result_list:
                if cond(k) and v['GTs'] > 0:
                    cond_met_map.append(v['mAP'])
            cond_met_map = np.mean(cond_met_map)
            report_dict[name] = cond_met_map
        return report_dict


def eval_map_flexible(det_results,
                      annotations,
                      match_thrs=[0.5],
                      breakdowns=[],
                      affinity_calculator=dict(type='LidarIOU3D'),
                      matcher=dict(type='MatcherCoCo'),
                      tp_metrics=[],
                      classes=None,
                      logger=None,
                      report_config=[
                          ('map', lambda x: x['breakdown'] == 'All')
                      ],
                      nproc=None):
    assert len(det_results) == len(annotations)

    if nproc is None:
        nproc = 0
    elif nproc < 0:
        nproc = cpu_count() or 0

    fse = FlexibleStatisticsEval(classes, match_thrs, breakdowns,
                                 affinity_calculator, matcher, tp_metrics,
                                 nproc)

    eval_result_list = fse.statistics_eval(det_results, annotations)
    report = fse.report(eval_result_list, report_config)

    table_data = [list(eval_result_list[0][0].keys()) + list(
        eval_result_list[0][1].keys())]
    for k, v in eval_result_list:
        table_data.append(
            [k['Class'], k['Breakdown'], k['Thres'],
             v['Dets'], v['GTs'], f'{100 * v["Recall"]:.3f}',
             f'{100 * v["mAP"]:.3f}'] + [f'{_v:.3f}' for _v in
                                         list(v.values())[4:]])
    table = AsciiTable(table_data)
    print_log('\n' + table.table, logger=logger)

    return report
