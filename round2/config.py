import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    data_path = "/home/tione/notebook/data"

    # ========================= Data Configs ==========================
    parser.add_argument('--labeled_annotation', type=str, default= data_path + '/annotations/labeled.json')
    parser.add_argument('--unlabeled_annotation', type=str, default= data_path + '/annotations/unlabeled_new.json')
    parser.add_argument('--labeled_zip_feats', type=str, default= data_path + '/clip_features/labeled.zip')
    parser.add_argument('--unlabeled_zip_feats', type=str, default= data_path + '/clip_features/unlabeled.zip')
    
    parser.add_argument('--test_annotation', type=str, default= data_path + '')
    parser.add_argument('--test_zip_frames', type=str, default= data_path + '')
    parser.add_argument('--test_zip_feats', type=str, default=data_path + '/zip_feats/test_a.zip')
    
    parser.add_argument('--test_output_csv', type=str, default='./data/result.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=32, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=0, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='./save/')
    parser.add_argument('--pretrain_model_name', type=str, default='best')
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--ckpt_file_1', type=str, default=None)
    parser.add_argument('--ckpt_file_2', type=str, default=None)
    parser.add_argument('--best_score', default=0.6, type=float, help='save checkpoint if mean_f1 > best_score')
    parser.add_argument('--best_loss', default=100, type=float)

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    parser.add_argument('--max_steps', default=100000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=8e-5, type=float, help='initial learning rate')
    parser.add_argument('--clip_learning_rate', default=2e-5, type=float, help='initial clip learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--val_steps', default=750, type=int)

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='/home/tione/notebook/env/pretrain/chinese-macbert-base')
    parser.add_argument('--bert_seq_length', type=int, default=128)
    parser.add_argument('--bert_learning_rate', type=float, default=3e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)
    
    parser.add_argument('--visual_pretrained_path', type=str, default='./clip_ckpt_file/ViT-B-32.pt') # 改过名字
    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=512)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')
    
    

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")

    # ========================== Pretrain =================================
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    
    parser.add_argument('--use_lxrt', action='store_true')
    
    return parser.parse_args()


args = parse_args()

