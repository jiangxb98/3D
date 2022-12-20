dataset_type = 'CowaSegDataset'
sensor_signals = ['x', 'y', 'z', 'intensity']
sensors = [0]
point_radius_range = [1.5, 50]
point_cloud_range = [-50.0, -50.0, -2.0, 50.0, 50.0, 4.0]
# class_names = (
#     'unlabeled', 'road', 'sidewalk', 'curb', 'other-ground', 'terrain',
#     'vegetation', 'trunk', 'framework', 'building', 'fence', 'pole',
#     'traffic-sign', 'other-structure', 'noise', 'road-users', 'road-block')

# road-plane includes road and sidewalk, pillars includes trunk and pole
class_names = (
    'unlabeled', 'road-plane', 'curb', 'other-ground', 'terrain',
    'vegetation', 'pillars', 'framework', 'building', 'fence',
    'traffic-sign', 'other-structure', 'noise', 'road-users', 'road-block')
map_labels = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 10, 11, 12, 13, 14]
input_modality = dict(use_lidar=True, use_camera=False)

old_info_path='clips'
old_datainfo_client_args = dict(
    backend='MONGODB',
    database='cowa3d-base',
    host='mongodb://root:root@172.16.110.100:27017/')

train_info_path = 'training/infos' # collection name
val_info_path = 'validation/infos'
datainfo_client_args = dict(
    backend='MONGODB',
    database='cowa3d-ruby-vel',
    host='mongodb://root:root@172.16.110.100:27017/')

pts_client_args = dict(
    backend='MINIO',
    bucket='cowa3d-base',
    endpoint='oss01-api.cowadns.com:30009',
    secure=False)

labels_client_args = dict(
    backend='MINIO',
    bucket='cowa3d-seg',
    endpoint='oss01-api.cowadns.com:30009',
    secure=False)

info_path_x3 = 'sampled-clips'
datainfo_client_args_x3 = dict(
    backend='MONGODB',
    database='cowa3d-x3-voted-data',
    host='mongodb://root:root@172.16.110.100:27017/')

pts_client_args_x3 = dict(
    backend='MINIO',
    bucket='cowa3d-x3-voted-data',
    endpoint='oss01-api.cowadns.com:30009',
    secure=False)

labels_client_args_x3 = dict(
    backend='MINIO',
    bucket='cowa3d-x3-voted-label',
    endpoint='oss01-api.cowadns.com:30009',
    secure=False)


train_parts = ['69e604a410044158c4afade2d036eca7',
               '5fdadc98cd5f67d4d7e4ad891cb48d1c',
               '75345c5759050c7a13519b874f8cf6e3',
               '17fa989a555b5f0510ee305fbb012e42',
               '9927cf6ec4b6dc46c26e9aa6dc758a67',
               '31319a0fedd36a87c4c9e5426b1f8768',
               '035fa1317ef6dca0cb69fcddb81d5e07',
               'fa29b7d0d753210f1e0e09e770d62f8b',
               'af42a27c576b04577e47382e90718e83',
               '36b40e99d6fc102205596510f3641fcc',
               '0760824c4b592c84d8e4fee0c0ee8624',
               'c8eb1fed79bad89ca54a1453e5bde86a',
               'b88f91d2636942f0442b8bb3a5ad0cf8',
               '46b56df648432cd1bb693d8d7fda9ead',
               'ce787110de9dc6c7f6ace2d1c1262d85',
               'e986ca27f8172f44d0cee7784e241ae1',
               'b802150d6cc5ee1d8c2977db48f1aa63',
               'dc07d0b83022badca2cf758e5590424f',
               '764cfe34ca6610dfadf6e462f233c13e',
               '39847ec6d40fc1c04098e4f7f4f58448',
               '3acd3dbbabceb7523d96745bb332eb68',
               '9deacd93dbc9b7072a8401dfc9717d38',
               '6f498d6ea0f45088772ec8d4aa41b825',
               '4f7c1aa2d3b5d5e55fac62d9699f5cf2',
               '47b4862ba9fdca1e0f25980d7c92de45',
               '2886200735e1f12f014d75ee86471fec',
               'c0cf64da45cd43039bb0d8c0f1b3f178',
               '945209c930eb4f9b216730fa4bff72d5',
               '2eb271707393934607d9980c5a92774e',
               'bdab97bb57f962b9eb35148a5cb86625',
               'fc258781bc2150a6b2c94185c260b945',
               '895afdfd32ad7e95e746eef2248603a4',
               '4f3c4bb511896004a1b02579a84d937d',
               'b17f90c7093fd6a0cd3ae5617f96bf92',
               '1df74e6b4b4ec010d4278d1a2835191d',
               '8beb72f41697903b996ae8e0f9f57d03',
               '0e314557c39835a74e2ff6c7b80bd11e',
               'cb272250efd733346bda7ff4213c43a4',
               '0f71dff7b2637d715e95fb3c6883fd50',
               'd9d2c997b69f5ca6bbc90bdfd7e2715b',
               '93db318b876d1acf27138b9c10e71d9a',
               'ebc6e430629b76b7e8c81df5fe8c144d',
               'c7f6dfb2ceeafe9807cdea1ab436568f',
               '19a81c1e56c7e86b93c3b43833a996f1',
               '3f1bdaed9173014467bf963f0250214a',
               'b60c051e00c8ffd547c26e34b3893002',
               '7549d38814dcfcbf79843385497cc535',
               '44d9a445550f9ee1f6ae92ae5333bdc2',
               '218741f703a571e61daca68253c174ce',
               '9df2fd88a3a62a83f2b1610c270347b9',
               '8d45e00c6a6b075609690bd4e557dbb9',
               'ea297910427e4aa05e408a8c733c04ba',
               '151468275bd1359f4cdf5a921f72afe5',
               '12cf4b121198752460a074f356d8293d',
               'cb3151bd82dd6d5345c0ee7e6a66b144',
               '772314e71596399b5f0be8db1581c875',
               '7baa5dd98276c907ff136e6e75ed99de',
               '3207a0bfcfd0b833fd658631b7be9e47',
               '3d42cde4edcba23f2bc9bd34ed2d5899',
               '3c5a60a60314537b2d49d9354776e051',
               '427ef3b48535ec72218c3882e461ba3d',
               '723705c159db8fd62d8f5305c4a4f1bc',
               '17e9a19ff28365407c0dc502437d411e',
               '61fcaefe74e5464e344fce48dc677d5f',
               'a143af87cde2a5bea8f3a98573bf1836',
               'f4c44c7d265d36dad5d0a0d352f08a50',
               '2fa7ba55c9b11ae5b54a7c0b930f4820',
               'c7be91c6e850a316415b08789a985c92',
               '481fa94636bc7a4c95e9188f40ef9fc8',
               '4e6a54a3cc3bd88d7ba2a925939cc73c',
               'ec7383b95572831c593d524ae70a891b',
               '560e9d28e7bee905d6778c10a121d0e4',
               '88bdf4fbd41d98184767f0625ae172d8',
               '8af22d0d39257ad5ed0cb5347b06fd47',
               '784d064a8a5938b3e767f3fccc97ae10',
               'bc09c506ca57e28763ee1025831497fb',
               '743f777456326a33d778a13e7bd4307f',
               '8d7a3b5af86e0ec02979d6a4b2172008',
               '76696aa5ed537b599a072fbf20d1127e',
               '1540a1518d14e2ba926b9fcf673dfe03',
               'e7265955de95dfd1091e7f9b3a6ccb60',
               'ee73df799a43bb2d3236e17251cd4788',
               'd6378c8436f0f4c3b72611bc389f5a54',
               '23dea24a5ea0cee87951361b3729e8ec',
               'd25122f3175fc3b0d82a74c4aa2f8d6d',
               '2f635c05c7aad8f4d0680ac4b954be23',
               'c8983fcdb982cedf960b02d8bfe6811f',
               'fa1718d9bba9cecaf3c0a01c7b3baf70',
               '4433f866fc1a549afd073e6c567fda76',
               'bc58d4f3405b7e93a62dd71141490241',
               'db9afbcc9d8b69c92a4acf39656a305b',
               '4b8737dad2cf681eb0f5705298df9d01',
               '555bb0e85d316f9513f6ccc7a3225f91',
               '99e711203fd9aac23805b1d227f4770f',
               '4b2d7f6bcd4ae1e25d9c9a51ba13ca75',
               '8e9047ad7c8eb2be921c779b5a872369',
               '7699deb50e982a7bef5db505810038b4',
               'fbbe447e06f6c719fbc4e3c51b2cd14d',
               '5568be7943e5153aa03d4bc4debb7c6b',
               'e2f47fd5739089e66fb2875885537959',
               '1469a360bd84c5bf57ae945840d06049',
               'd376f1cae0a8425fc5a60b327a43030c',
               'e3ebb25ebda7e2b01479a9949279a703',
               'ef85cc4039a05570f71b033f75cd2833',
               '2b1f54963e3fc41ee29b214633c668cf',
               '3542bc754475f1f6e17f598c4beaf86f',
               'f9522f9a7fac4702d7bda52cd0927c48',
               'a72971a0e619d8fa27656752f4e70305',
               'bc100f14d26aa1841f9ca05135df7f7a',
               '3e0f277985f884ebbc0d0fb06782c1e3',
               '6f35567fbcc7bb7bac0baa24d117308e',
               '7c4bc73793071ae4ba02003a549c01cc',
               '474f3385b5dec8836b8a649e613cd28a',
               '02d35543180bef5c4228bc6c3e61df10',
               '5601bdcad23b8c2c488c0c8c100ccf31',
               '11abc205a1d00665d8e46a1207bf3d44',
               '527975c847f8444ab1b0dab767d3a76b',
               'e67f903aa19b70d3a8ce7473d207abc2',
               '6c66042006a69cbd1b83cd5e04675085',
               '6f793c67ff8872e0d8e97a0822a5f1fb',
               '587c5c7cff1bbd6a5294398e54ec1dee',
               '90c2c5fe76e11795f8b684e6bcacb625',
               'bf54c49bcc1f682fb6ba49266c08ce46',
               '477792bd0e32ca8c0cf7aef2dabecdea',
               '7808431d8f69a9d46f060638bdec4e67',
               '68aebff72a6c3a0100df614d418ffae2',
               'b0861dabb43ac6ba6f25e3a7f1bfaee5',
               '4ccbecdbcc0b4e7c633c2d04559b23ab',
               '81d99baff757baf3e06e80ead357a8fc',
               '66651c9c071c0e55c9e438cc420f54dc',
               '4cbe25e117c484dd30566400690ad8ce',
               '1fbaaa64fdd2cc95a03b4695b53a61ac',
               '13566eff26f358ccfab51e23b4a3568b',
               'f2219808405fb42c9bee1e73fbc6b1db',
               'b867d8e6b4f9cef45548a2922b58aa4c',
               '20bdd7de69ff0ff4b2ed17de1aa5fd68',
               '723e916b453d801b18019bd81895827e',
               'c17c68c898c90ca5937e0ed5c89d8e5d',
               'f9971fd61fea6f2800cbddbe16e253ef',
               'e095b2a796a5e71b959d294b194de0b6',
               '550e7e7323f5392eaeee431f6dfc8e5f',
               '397932096f44cfb7b696f21d2f4a2363',
               '11b755fe4b64f307aea506aa608751b4',
               '903e57797521ad7e53693f7c5f4cc0ae',
               'f69b1ebf6c729a39cbdda8146df3fe8f',
               '1e4ec5eec6efb97e00d89e049d625d7c',
               '26a651f9908808ef973e59f1125983ee',
               '117933b0b6c5276beb098b21b8b988f4',
               '467fc385cbe923b08612edacfd2872f9',
               '254e01d44e3c65edf7ef5290e177c95b',
               'fa24ba0ee6a289bed7361c5119853f8e',
               'c3d9f05999c1998d76a702d77cf132b4',
               '108e8330f8fc20d2e1e8c83f394167fc',
               '3e83fbffa7b5cdd38e716ff16508ff5b',
               'eb390c6e51bce9e01e569c33b20cdd28',
               'd47356ed221eb7383acbf6bbd15fe19a',
               '7bf6bfb2eae8a3ab17925501e5945a72',
               '2ae3908faebaa3f71b697f69e001d2d1',
               'e23aec3873cc4b943ae68664d45ada0e',
               '0ae7b8275e3e8bded2b1a9908667aa6e',
               '15f2cecb51c34a4315e3fc984d96e139',
               'a9279ea48131efde5dc840aba9c7a692',
               '6817e3b2b496d98f2ab3c59308545b8a']

train_parts_x3 = ["x3_116fb60e261b5ada8e98efe183c7b7e7",
               "x3_fcc515187c5857d192dbbee39da20b73",
               "x3_e612d8daa47a5f3eb1768c42a75c2a12",
               "x3_67d8eea7233850dd8caa4961fb8cf29e",
               "x3_06138f8bb24f5dce86a6a5c67f1b342c",
               "x3_b88a66721faa5a9d8b98b71a7a7ddef6",
               "x3_710ef9becf87510b9438a58672ebb9c8",
               "x3_b07ffd4a82f25959a1f399ed87dce372",
               "x3_4dd7c472cad65c2db164c1c61eb898cf",
               "x3_ca466d0f295950feb9072d5536e9eb2f",
               "x3_ec0e6415dad75e119228f6f845d6bac0",
               "x3_4bc84e8ee02c526ea4390ed6cf7017c7",
               "x3_3c2e434b1f8351b4a52003599edb9aa9",
               "x3_caf66a6485a75ed99a9c360d1b3a87e4",
               "x3_8120007d11375a548d756ad09aaeb89e",
               "x3_d79f337c95ab5cfcba80b1ea0d40f78e",
               "x3_a77aeb2d745c5cc59a970336481deecb",
               "x3_fb74112f3be25572b2cf1f355f342057",
               "x3_c30af5d3e1bd5a06b90a9fb4ec5a6abd",
               "x3_0e6f37b5b9f65edbbd3e83e28fc99fa0",
               "x3_c8e7ac651fb35ed5823172a529ee30ae",
               "x3_c741a62a09f952ea85ffa1ebb50b9691",
               "x3_b49bec3dd4e8544292ba4e6578435bef",
               "x3_144929b5dd245f8fbca26424ef5b4ab3",
               "x3_c3b3e6f251f95151820ea9e26dc4964a",
               "x3_455cf0226ed550e9b0cd4be85ca346d6",
               "x3_4feaf627d468567da6ea1deecd5e13dd",
               "x3_87843b6f6c985d9595f67ef315c900dd",
               "x3_b227f9e11c6e52b7afda922d982ddd45",
               "x3_cd236bc29d955696bd86bed110755528",
               "x3_22656f26d63e574d848987b406b84663",
               "x3_4408a290b6d6523b9fa88661f9561256",
               "x3_dfd61d36105354288e23f35f5bcbce68",
               "x3_681e9be6ea555217845e541a2d796c6f",
               "x3_e2d78bfdf1a45509b1ccae357e14c958",
               "x3_bab4444bfddf58babfa8b9d024a2bcce",
               "x3_1629c19e79d95165afb38ca82dd60a7e",
               "x3_2d531bc8740557acbbd9ed5ea09d06ec",
               "x3_0dd220a37e945d08836a3d303d158b09",
               "x3_946b5422af575daa83041dbd3f525fed",
               "x3_f8ecd6994234557892cccbf78bcddc13",
               "x3_37017da7dba9545cb47cb30e3114b9c3",
               "x3_0b06e2d5d5155049b2a0f1caad5a7cc4",
               "x3_e47dc76f73c25f17b575a58059b92dec",
               "x3_6ea2906d57ba53e984e1be796524584b",
               "x3_3a8892c90af9541c98c4475d04c4f546",
               "x3_014245bec2965d9b91c83a9c0e775378",
               "x3_62e31b3976325626bba4a9f89ca0add7",
               "x3_98d79422049f55cb814cda42344edb74",
               "x3_c47aebe04b165ce19008c12e2c827ba3",
               "x3_d55118d6588150e4a2863225577b1697",
               "x3_9958b7b25df55b9c85237536adc1c97a",
               "x3_0ceaa6c095cb5c40b36ad23cb9194dfb",
               "x3_c550165bbd385172ba03e8d6aeb3cbed",
               "x3_2c6cabacda4f5fdc8b56bfc693858cbc",
               "x3_58b43992e8a15655a267e6c13b4040b9",
               "x3_0cae3dc71ae65baf9dfe81b4dc594ab2",
               "x3_838fbbfbd8865b65b6531be6cdc5ac9f",
               "x3_ccec22946ce55389b7039d28ff962131",
               "x3_dc27b1456ff25781b01c4da5cbbf69f2",
               "x3_166cb7c2097953ed851482470bfaf925",
               "x3_5676cede309b5fafb7c9a1161f2f677d"]

val_parts = ['0d0cc7b686ed3188ad73d3339e1fbf62',
             '17c07641a947b2915aa33ab6d34da75b',
             '1ea59aed67583eb0e128cc817d620e9a',
             '2e030d40f51cd0efaab8b24ea95630bc',
             '61d30701c7a5cf56713c2a640afef09b',
             '80ec11d2b9184af58596b454c9bf21f1',
             '8a11854ca1d677b94ac0d68d6a19cfea',
             '917935f080e6c6d9b9456413233ea70a',
             '971a94b057179621f6b321f70f675869',
             '9b68cc1ecbcdc978f29cb595d2492006',
             'e22847460bb6201fbc5b63ed40d09049',
             'ea35bd3ed485efb935e113a8d18be050']

val_parts_x3 = ["x3_da7faf076c49580b84e19078497f039b",
             "x3_e1a73538ffb05cea899118eb85fddc61"]


train_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=pts_client_args),
    dict(
        type='LoadSweeps',
        sweeps_num=1,
        coord_type='LIDAR',
        file_client_args=pts_client_args),
    dict(
        type='LoadSegAnnos3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        file_client_args=labels_client_args),
    dict(type='RandomFlip3D',
         flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1]),
    dict(type='PointShuffle'),
    # dict(type='PointsRadiusRangeFilter',
    #      point_radius_range=point_radius_range),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=pts_client_args),
    dict(
        type='LoadSweeps',
        sweeps_num=1,
        coord_type='LIDAR',
        file_client_args=pts_client_args),
    dict(
        type='LoadSegAnnos3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        file_client_args=labels_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            # dict(type='PointsRadiusRangeFilter',
            #      point_radius_range=point_radius_range),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=pts_client_args),
    dict(
        type='LoadSweeps',
        sweeps_num=1,
        coord_type='LIDAR',
        file_client_args=pts_client_args),
    dict(
        type='LoadSegAnnos3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        file_client_args=labels_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            # dict(type='PointsRadiusRangeFilter',
            #      point_radius_range=point_radius_range),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
        ])
]

train_pipeline_x3 = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=pts_client_args_x3),
    dict(
        type='LoadSegAnnos3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        file_client_args=labels_client_args_x3),
    dict(type='RandomFlip3D',
         flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1]),
    dict(type='PointShuffle'),
    # dict(type='PointsRadiusRangeFilter',
    #      point_radius_range=point_radius_range),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
]
test_pipeline_x3 = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=pts_client_args_x3),
    dict(
        type='LoadSegAnnos3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        file_client_args=labels_client_args_x3),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            # dict(type='PointsRadiusRangeFilter',
            #      point_radius_range=point_radius_range),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline_x3 = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=pts_client_args_x3),
    dict(
        type='LoadSegAnnos3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        file_client_args=labels_client_args_x3),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            # dict(type='PointsRadiusRangeFilter',
            #      point_radius_range=point_radius_range),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            info_path=train_info_path,
            parts=train_parts+val_parts,
            sensors=sensors,
            sensor_signals=sensor_signals,
            map_labels=map_labels,
            filter_empty_gt=False,
            pcd_limit_range=point_cloud_range,
            task='seg',
            pipeline=train_pipeline,
            det_classes=None,
            seg_classes=class_names,
            test_mode=False,
            modality=input_modality,
            box_type_3d='LiDAR',
            load_interval=1,
            datainfo_client_args=datainfo_client_args)),
    val=dict(
        type=dataset_type,
            info_path=val_info_path,
            parts=train_parts+val_parts,
            sensors=sensors,
            sensor_signals=sensor_signals,
            map_labels=map_labels,
            filter_empty_gt=False,
            pcd_limit_range=point_cloud_range,
            task='seg',
            pipeline=eval_pipeline,
            det_classes=None,
            seg_classes=class_names,
            test_mode=True,
            modality=input_modality,
            box_type_3d='LiDAR',
            load_interval=1,
            datainfo_client_args=datainfo_client_args),
    test=dict(
        type=dataset_type,
            info_path=val_info_path,
            parts=train_parts+val_parts,
            sensors=sensors,
            sensor_signals=sensor_signals,
            map_labels=map_labels,
            filter_empty_gt=False,
            pcd_limit_range=point_cloud_range,
            task='seg',
            pipeline=test_pipeline,
            det_classes=None,
            seg_classes=class_names,
            test_mode=True,
            modality=input_modality,
            box_type_3d='LiDAR',
            load_interval=1,
            datainfo_client_args=datainfo_client_args))

evaluation = dict(interval=1, pipeline=eval_pipeline)
