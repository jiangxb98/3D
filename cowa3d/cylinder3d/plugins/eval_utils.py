def pts_semantic_confusion_matrix(pts_pred, pts_target, num_classes):
    pts_cond = pts_target * num_classes + pts_pred
    pts_cond_count = pts_cond.bincount(minlength=num_classes * num_classes)
    return pts_cond_count[:num_classes * num_classes].reshape(
        num_classes, num_classes).cpu().numpy()
