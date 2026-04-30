import argparse
import datetime
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pickle
import time
import random
import numpy as np
import torch
from model import NET
from model_rexpert import REXPERTNET
from utils import *
from tqdm import tqdm
from difffu_21 import DiffuRec


"""
seed = 987
np.random.seed(seed)
torch.manual_seed(seed)
"""
seed = 987
np.random.seed(seed)
torch.manual_seed(seed)

def split_by_time(data):
    times = []
    latest_t = -1
    for i, tuple in enumerate(data):
        t = tuple[3]
        if t != latest_t:
            times.append([i, t])
            latest_t = t
    return times



def load_pickle_array(path, keep_sparse=True):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if keep_sparse:
        return obj
    if hasattr(obj, 'toarray'):
        return obj.toarray()
    return obj

def create_model_diffu(args):
    diffu_pre = DiffuRec(args)
    return diffu_pre

def make_backgrounds(data, data_t, k):
    backgrounds = []
    times = data_t[-k:]
    for i in range(len(times)):
        l = times[i][0]
        if i < len(times) - 1:
            r = times[i + 1][0]
        else:
            r = len(data)
        backgrounds.append(np.array(data[l:r][:, :3]))
    return backgrounds


def train(args):
    settings = {}

    num_nodes, num_rels, num_t = get_total_number('./data/' + args.dataset, 'stat.txt')
    train_data, _ = load_quadruples('./data/' + args.dataset, 'train.txt')
    try:
        dev_data, _ = load_quadruples('./data/' + args.dataset, 'valid.txt')
    except:
        print(args.dataset, 'does not have valid set.')
    test_data, _ = load_quadruples('./data/' + args.dataset, 'test.txt')
    try:
        total_data, _ = load_quadruples('./data/' + args.dataset, 'train.txt', 'valid.txt', 'test.txt')
        valid_filter_data, _ = load_quadruples('./data/' + args.dataset, 'train.txt', 'valid.txt')
    except:
        total_data, _ = load_quadruples('./data/' + args.dataset, 'train.txt', 'test.txt')
        valid_filter_data = total_data
        print(args.dataset, 'does not have valid set.')

    train_t = split_by_time(train_data)
    if args.use_valid:
        dev_t = split_by_time(dev_data)
    else:
        dev_t = []
    test_t = split_by_time(test_data)

    train_s_frequency_f = '/train_s_frequency.txt'
    train_o_frequency_f = '/train_o_frequency.txt'
    train_s_label_f = '/train_s_label.txt'
    train_o_label_f = '/train_o_label.txt'

    dev_s_frequency_f = '/dev_s_frequency.txt'
    dev_o_frequency_f = '/dev_o_frequency.txt'
    dev_s_label_f = '/dev_s_label.txt'
    dev_o_label_f = '/dev_o_label.txt'

    test_s_frequency_f = '/test_s_frequency.txt'
    test_o_frequency_f = '/test_o_frequency.txt'
    test_s_frequency_offline_f = '/test_s_frequency_offline.txt'
    test_o_frequency_offline_f = '/test_o_frequency_offline.txt'
    test_s_label_f = '/test_s_label.txt'
    test_o_label_f = '/test_o_label.txt'

    train_s_frequency = load_pickle_array('./data/' + args.dataset + train_s_frequency_f, keep_sparse=True)
    train_o_frequency = load_pickle_array('./data/' + args.dataset + train_o_frequency_f, keep_sparse=True)
    train_s_label = load_pickle_array('./data/' + args.dataset + train_s_label_f, keep_sparse=True)
    train_o_label = load_pickle_array('./data/' + args.dataset + train_o_label_f, keep_sparse=True)

    if args.use_valid:
        dev_s_frequency = load_pickle_array('./data/' + args.dataset + dev_s_frequency_f, keep_sparse=True)
        dev_o_frequency = load_pickle_array('./data/' + args.dataset + dev_o_frequency_f, keep_sparse=True)
        dev_s_label = load_pickle_array('./data/' + args.dataset + dev_s_label_f, keep_sparse=True)
        dev_o_label = load_pickle_array('./data/' + args.dataset + dev_o_label_f, keep_sparse=True)

    if args.mode == "online":
        test_s_frequency = load_pickle_array('./data/' + args.dataset + test_s_frequency_f, keep_sparse=True)
        test_o_frequency = load_pickle_array('./data/' + args.dataset + test_o_frequency_f, keep_sparse=True)
        test_s_label = load_pickle_array('./data/' + args.dataset + test_s_label_f, keep_sparse=True)
        test_o_label = load_pickle_array('./data/' + args.dataset + test_o_label_f, keep_sparse=True)
    elif args.mode == "offline":
        test_s_frequency = load_pickle_array('./data/' + args.dataset + test_s_frequency_offline_f, keep_sparse=True)
        test_o_frequency = load_pickle_array('./data/' + args.dataset + test_o_frequency_offline_f, keep_sparse=True)
        test_s_label = load_pickle_array('./data/' + args.dataset + test_s_label_f, keep_sparse=True)
        test_o_label = load_pickle_array('./data/' + args.dataset + test_o_label_f, keep_sparse=True)
    else:
        print("Invalid mode!")
        exit()

    device = args.device
    diffu_rec = None
    if args.model_name.lower() == 'dmsa':
        diffu_rec = create_model_diffu(args)
        model = NET(num_nodes, num_rels, num_t, diffu_rec, args)
    elif args.model_name.lower() == 'rexpert':
        model = REXPERTNET(num_nodes, num_rels, num_t, args)
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step)
    model = model.to(device)
    now = datetime.datetime.now()
    dt_string = args.description + now.strftime("%d-%m-%Y,%H-%M-%S") + \
                args.dataset + '-EPOCH' + str(args.max_epochs)
    main_dirName = os.path.join(args.save_dir, args.dataset, dt_string)
    if not os.path.exists(main_dirName):
        os.makedirs(main_dirName)

    model_path = os.path.join(main_dirName, 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    settings['main_dirName'] = main_dirName
    file_training = open(os.path.join(main_dirName, "training_record.txt"), "w")
    file_training.write("Training Configuration: \n")
    for key in settings:
        file_training.write(key + ': ' + str(settings[key]) + '\n')
    for arg in vars(args):
        file_training.write(arg + ': ' + str(getattr(args, arg)) + '\n')
    print("Start training...")
    file_training.write("Training Start \n")
    file_training.write("===============================\n")

    train_backgrounds, dev_backgrounds, test_backgrounds = [], [], []
    if args.use_valid:
        dev_backgrounds = make_backgrounds(train_data, train_t, args.history_len)
        test_backgrounds = make_backgrounds(dev_data, dev_t, args.history_len)
    else:
        test_backgrounds = make_backgrounds(train_data, train_t, args.history_len)

    train_his_g, dev_his_g, test_his_g = [], [], []
    if args.use_valid:
        dev_his_g = [get_big_graph(bg, num_nodes, num_rels).to(device) for bg in dev_backgrounds]
    test_his_g = [get_big_graph(bg, num_nodes, num_rels).to(device) for bg in test_backgrounds]

    train_list = split_by_time3(train_data)
    dev_list = split_by_time3(dev_data)
    test_list = split_by_time3(test_data)
    err_mat_list = []
    for snap in train_list + dev_list + test_list:
    # for snap in train_list + test_list:
        err_mat_list.append(get_entity_related_relation(snap, num_nodes, num_rels))  # 365 [7128, 460]
    # err_mat = sparse_mx_to_torch_sparse_tensor(err_mat_list[i])

    train_inv_list = get_inverse(train_list, num_rels)
    dev_inv_list = get_inverse(dev_list, num_rels)
    test_inv_list = get_inverse(test_list, num_rels)
    err_inv_mat_list = []
    for snap in train_inv_list + dev_inv_list + test_inv_list:
    # for snap in train_inv_list + test_inv_list:
        err_inv_mat_list.append(get_entity_related_relation(snap, num_nodes, num_rels))

    # print('Length of time:'+str(len(train_t+dev_t+test_t)))
    epoch = 0
    best_valid_hits1 = float('-inf')
    best_valid_metrics = None
    best_epoch = 0
    while epoch < args.max_epochs:
        model.train()
        epoch += 1
        print('$Start Epoch: ', epoch)
        file_training.write('$Start Epoch: ' + str(epoch) + '\n')
        loss_epoch = 0
        optimizer.zero_grad(set_to_none=True)
        time_begin = time.time()
        _batch = 0
        list_diff_data = []
        for batch_data in tqdm(
                make_batch4(train_data, train_s_frequency, train_o_frequency, train_s_label, train_o_label, train_t,
                           args.batch_size, err_mat_list, err_inv_mat_list), total=len(train_t), dynamic_ncols=True):
            triples = np.asarray(batch_data[0][:, :3])
            batch_data[0] = torch.from_numpy(batch_data[0])
            batch_data[1] = torch.from_numpy(batch_data[1]).float()
            batch_data[2] = torch.from_numpy(batch_data[2]).float()
            batch_data[3] = torch.from_numpy(batch_data[3]).float()
            batch_data[4] = torch.from_numpy(batch_data[4]).float()

            batch_data[0] = batch_data[0].to(device)
            batch_data[1] = batch_data[1].to(device)
            batch_data[2] = batch_data[2].to(device)
            batch_data[3] = batch_data[3].to(device)
            batch_data[4] = batch_data[4].to(device)

            if list_diff_data == []:
                list_diff_data.append(batch_data[0])

                g = get_big_graph(triples, num_nodes, num_rels)
                if len(train_backgrounds) >= args.history_len:
                    train_backgrounds = train_backgrounds[1:]
                    train_his_g = train_his_g[1:]

                train_backgrounds.append(triples)
                g = g.to(device)

                train_his_g.append(g)
                continue
            else:
                batch_loss = model(batch_data, train_his_g, list_diff_data, 'Training')
            if batch_loss is not None:
                error = batch_loss
            else:
                continue
            error.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, 'clear_runtime_cache'):
                model.clear_runtime_cache()
            loss_epoch += error.item()
            _batch += 1

            list_diff_data.append(batch_data[0])
            list_diff_data = list_diff_data[-args.max_len:]

            g = get_big_graph(triples, num_nodes, num_rels)
            if len(train_backgrounds) >= args.history_len:
                train_backgrounds = train_backgrounds[1:]
                train_his_g = train_his_g[1:]

            train_backgrounds.append(triples)
            g = g.to(device)

            train_his_g.append(g)

        if hasattr(model, 'clear_runtime_cache'):
            model.clear_runtime_cache()
        if torch.cuda.is_available() and str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        scheduler.step()
        epoch_time = time.time()
        print("Done\nEpoch {:04d} | Loss {:.4f}| time {:.4f}".
              format(epoch, loss_epoch / _batch, epoch_time - time_begin))
        file_training.write("******\nEpoch {:04d} | Loss {:.4f}| time {:.4f}".
                            format(epoch, loss_epoch / _batch, epoch_time - time_begin) + '\n')
        # Validation: default to validating on the test split every N epochs.
        # If --validate_split valid is chosen and a valid set is available, validate on valid.
        if epoch % args.validate_every == 0:
            print("Start valid...")
            file_training.write("Start valid...\n")
            use_dev = (args.validate_split == "valid") and args.use_valid
            if use_dev:
                valid_metrics, _, _, _, _, _, _ = execute_valid_hits1(
                    args, dev_backgrounds.copy(), dev_his_g.copy(), num_nodes, num_rels, valid_filter_data, model,
                    dev_data, dev_s_frequency, dev_o_frequency, dev_s_label, dev_o_label, dev_t,
                    err_mat_list[len(train_t):], err_inv_mat_list[len(train_t):], train_data, train_t
                )
            else:
                # validate on test split (default)
                offset = len(train_t) + (len(dev_t) if args.use_valid else 0)
                valid_metrics, _, _, _, _, _, _ = execute_valid_hits1(
                    args, test_backgrounds.copy(), test_his_g.copy(), num_nodes, num_rels, total_data, model,
                    test_data, test_s_frequency, test_o_frequency, test_s_label, test_o_label, test_t,
                    err_mat_list[offset:], err_inv_mat_list[offset:], train_data, train_t
                )
            valid_hits1 = valid_metrics['all_hits@1']
            print()
            print("Valid ALL Hits@1:", valid_hits1)
            print("Valid ALL MRR:", valid_metrics['all_mrr'])
            print("Valid Subject Recall@C:", valid_metrics['subject_recall@c'])
            print("Valid Object Recall@C:", valid_metrics['object_recall@c'])
            print("Valid ALL Recall@C:", valid_metrics['all_recall@c'])
            file_training.write("Valid ALL Hits@1:" + str(valid_hits1) + "\n")
            file_training.write("Valid ALL MRR:" + str(valid_metrics['all_mrr']) + "\n")
            file_training.write("Valid Subject Recall@C:" + str(valid_metrics['subject_recall@c']) + "\n")
            file_training.write("Valid Object Recall@C:" + str(valid_metrics['object_recall@c']) + "\n")
            file_training.write("Valid ALL Recall@C:" + str(valid_metrics['all_recall@c']) + "\n")
            if valid_hits1 > best_valid_hits1:
                best_valid_hits1 = valid_hits1
                best_valid_metrics = valid_metrics
                best_epoch = epoch
                torch.save(model, model_path + '/' + args.dataset + '_best.pth')
            if hasattr(model, 'clear_runtime_cache'):
                model.clear_runtime_cache()
            if torch.cuda.is_available() and str(device).startswith('cuda'):
                torch.cuda.empty_cache()
    print("Training done")
    file_training.write("Training done")
    file_training.close()

    # Evaluation
    if args.only_eva:
        dt_string = args.model_dir
        main_dirName = os.path.join(args.save_dir, dt_string)
        model_path = os.path.join(main_dirName, 'models')
        settings['main_dirName'] = main_dirName
    if args.filtering:
        if args.only_eva:
            file_test_path = os.path.join(main_dirName, "test_record_filtering_eva.txt")
        else:
            file_test_path = os.path.join(main_dirName, "test_record_filtering.txt")
    else:
        if args.only_eva:
            file_test_path = os.path.join(main_dirName, "test_record_raw_eva.txt")
        else:
            file_test_path = os.path.join(main_dirName, "test_record_raw.txt")

    file_test = open(file_test_path, "w")
    file_test.write("Testing starts: \n")
    if not os.path.exists(model_path + '/' + args.dataset + '_best.pth'):
        torch.save(model, model_path + '/' + args.dataset + '_best.pth')
    model = torch.load(model_path + '/' + args.dataset + '_best.pth')
    model.eval()
    model.args = args

    if args.use_valid:
        valid_metrics, _, _, _, _, _, _ = execute_valid_hits1(args, dev_backgrounds.copy(), dev_his_g.copy(), num_nodes, num_rels, valid_filter_data, model,
                                   dev_data, dev_s_frequency, dev_o_frequency, dev_s_label, dev_o_label, dev_t, err_mat_list[len(train_t):], err_inv_mat_list[len(train_t):], train_data, train_t)
        print("***Best Epoch:", best_epoch, "***Best Valid ALL Hits@1:", valid_metrics['all_hits@1'])
        print("***Best Valid ALL Recall@C:", valid_metrics['all_recall@c'])
        file_test.write("***Best Epoch: " + str(best_epoch) + " ***Best Valid ALL Hits@1: " + str(valid_metrics['all_hits@1']) + "\n")
        file_test.write("***Best Valid ALL Recall@C: " + str(valid_metrics['all_recall@c']) + "\n")
    s_ranks1, o_ranks1, all_ranks1, s_recall_c1, o_recall_c1, all_recall_c1 = execute_test5(args, test_backgrounds, test_his_g, num_nodes, num_rels, total_data,
                                                  model,
                                                  test_data, test_s_frequency, test_o_frequency, test_s_label,
                                                  test_o_label, test_t, err_mat_list[len(train_t+dev_t):], err_inv_mat_list[len(train_t+dev_t):], train_data, train_t)

    # evaluation for link prediction
    write2file(s_ranks1, o_ranks1, all_ranks1, file_test, s_recall_c1, o_recall_c1, all_recall_c1)
    file_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DMSA')
    parser.add_argument("--description", type=str, default='yago', help="description")
    parser.add_argument("-d", "--dataset", type=str, default='YAGO', help="dataset")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=20, help="maximum epochs")
    parser.add_argument("--valid-epochs", type=int, default=1, help="validation epochs")
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for nceloss")
    parser.add_argument("--lambdax", type=float, default=2.0, help="lambda")
    parser.add_argument("--history-len", type=int, default=1)
    parser.add_argument("--mode", type=str, default="offline")
    parser.add_argument("--graph-layer", type=int, default=2)

    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--lr-dc-step", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument("--filtering", type=str2bool, default=True)

    parser.add_argument("--only-eva", type=str2bool, default=False, help="whether only evaluation on test set")
    parser.add_argument("--use-valid", type=str2bool, default=False, help="whether using validation set")
    parser.add_argument("--model-dir", type=str, default="", help="model directory")
    parser.add_argument("--save-dir", type=str, default="result", help="save directory")
    parser.add_argument("--eva-dir", type=str, default="result", help="saved dir of the testing model")
    parser.add_argument("--gamma", type=float, default="0.1", help="gamma for score_enhanced")
    parser.add_argument("--timestamps", type=int, default="304", help="the total length of timestamps")
    parser.add_argument("--sample-len", type=int, default=3, help="sample length")

    parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
    parser.add_argument('--diffusion_steps', type=int, default=200, help='Diffusion step')
    parser.add_argument("--k_step", type=int, default=0, help="number of propagation rounds")
    parser.add_argument('--lambda_uncertainty', type=float, default=0.01, help='uncertainty weight')
    parser.add_argument('--noise_schedule', default='linear',
                        help='Beta generation')  ## cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt
    parser.add_argument('--rescale_timesteps', default=False, help='rescal timesteps')

    parser.add_argument('--scale', type=float, default=50, help='scale weight')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='beta_start weight')
    parser.add_argument('--beta_end', type=float, default=0.02, help='beta_end weight')
    parser.add_argument("--temperature_object",type=float, default=0.5)
    parser.add_argument("--pattern_noise_radio",type=float, default=1)
    parser.add_argument("--encoder_params", type=dict, default=None)
    parser.add_argument("--concat_con", action='store_true', default=True,
                        help="perform layer normalization ")
    parser.add_argument("--refinements_radio", type=float, default=0,
                        help="perform layer normalization ")
    parser.add_argument("--heads", type=int, default=4,
                        help="perform layer normalization")
    parser.add_argument('--emb_dropout', type=float, default=0.4, help='Dropout of embedding')
    parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of Transformer blocks')
    parser.add_argument('--num_blocks_cross', type=int, default=0, help='Number of Transformer blocks')
    parser.add_argument("--max_len", type=int, default=32)

    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file. Values override CLI defaults.")
    parser.add_argument("--model_name", type=str, default="dmsa", choices=["dmsa","rexpert"], help="Model to run.")
    parser.add_argument("--validate_every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--validate_split", type=str, default="test", choices=["test","valid"], help="Which split to use for validation (default: test).")

    # REXPERT (R²-DMSA) retrieval + reranking params
    parser.add_argument("--retrieval_topk", type=int, default=8, help="Top-K evidence events used after retrieval scoring (rexpert).")
    parser.add_argument("--retrieval_pool_size", type=int, default=256, help="Per-query recalled event pool size via structure recall (rexpert).")
    parser.add_argument("--history_pool_max", type=int, default=20000, help="Max flattened history events considered for retrieval (rexpert).")
    parser.add_argument("--retrieval_time_lambda", type=float, default=0.0, help="Time decay coefficient applied to retrieval attention logits (rexpert).")
    parser.add_argument("--retrieval_beta", type=float, default=0.0, help="Optional retrieval-based entity bias weight added to NCE logits (rexpert).")

    # Backward compatible alias: older configs used max_hist_steps for retrieval pool size.
    parser.add_argument("--max_hist_steps", type=int, default=64, help="(alias) older name for --retrieval_pool_size (rexpert).")

    # Candidate reranking params
    parser.add_argument("--rerank_topm", type=int, default=256, help="Top-M candidates from DMSA enhanced decoder to include in reranking (rexpert).")
    parser.add_argument("--rerank_max_cands", type=int, default=512, help="Final candidate budget for reranking (union of topM and retrieved) (rexpert).")
    parser.add_argument("--rerank_eta", type=float, default=1.0, help="Strength of reranker score delta added to logits (rexpert).")
    parser.add_argument("--lambda_rank", type=float, default=0.5, help="Weight of candidate-set reranking loss (rexpert).")
    parser.add_argument("--lambda_ret", type=float, default=0.0, help="Weight of retrieval contrastive loss (in-batch negatives) (rexpert).")
    parser.add_argument("--ret_tau", type=float, default=0.07, help="Temperature for retrieval contrastive loss (rexpert).")

    # Backward/compatibility alias: some configs used max_events to control rolling history length.
    parser.add_argument("--max_events", type=int, default=0, help="Alias for --max_len (kept for config compatibility).")

    args_main = parser.parse_args()
    print(args_main)

    # Load JSON config (overrides parsed args)
    if args_main.config is not None:
        with open(args_main.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args_main, k):
                setattr(args_main, k, v)
            else:
                print(f"[Config] Warning: unknown key '{k}' in config (ignored).")

    # Apply compatibility alias after config is loaded.
    # If a config provides max_events, treat it as max_len (rolling history buffer length).
    if getattr(args_main, "max_events", 0) and args_main.max_events > 0:
        args_main.max_len = int(args_main.max_events)

    # If an old config provides max_hist_steps, treat it as retrieval_pool_size unless explicitly set.
    if hasattr(args_main, "max_hist_steps") and hasattr(args_main, "retrieval_pool_size"):
        if int(args_main.retrieval_pool_size) == 256 and int(args_main.max_hist_steps) != 64:
            args_main.retrieval_pool_size = int(args_main.max_hist_steps)


    if not os.path.exists(args_main.save_dir):
        os.makedirs(args_main.save_dir)
    train(args_main)
