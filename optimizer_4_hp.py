#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/04/11 1:08
# @Author  : strawsyz
# @File    : optimizer_4_hp.py
# @desc:
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from print_color import print_red
import time

def append_commands(attr_name, values, commands, grid=False):
    global main_command
    if not grid:
        if values is not None and len(values) != 0:
            for value in values:
                command = f"{main_command} --{attr_name} {value}"
                commands.append(command)

        return commands
    else:
        new_commands = []
        for command in commands:
            if values is not None and len(values) != 0:
                for value in values:
                    new_command = f"{command} --{attr_name} {value}"
                    new_commands.append(new_command)

        return new_commands


def add_record(file_path, record):
    with open(file_path, mode="a") as f:
        f.write(record)


def run_command(command):
    p = os.popen(command)
    result = p.readlines()
    assert len(result) > 0
    if result[-1] == "\x1b[0m":
        result = result[-2]
    else:
        result = result[-1]
    record = f"{command}:\t{result}"
    return record  # 只将最后的结果返回


def select_free_gpu_by_memory(memory_threshold=None):
    import numpy as np
    command = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Free"
    p = os.popen(command)
    result = p.readlines()
    memory_gpu = [int(x.split()[2]) for x in result]
    free_gpu_id = np.argmax(memory_gpu)
    if memory_threshold is not None and memory_gpu[free_gpu_id] < memory_threshold:
        return None
    return str(free_gpu_id)


if __name__ == '__main__':
    # RECORD_FILEPATH = r"record.txt"
    DIR_PATH = "records"
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    RECORD_FILEPATH = fr"{DIR_PATH}/{int(time.time())}"
    # 用于解决一个error，不知道原因
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    parser = ArgumentParser(description='trail for network', formatter_class=ArgumentDefaultsHelpFormatter)
    # 固定不变的参数
    parser.add_argument('--OA', required=False, default="", help='other arguments')

    parser.add_argument('--LR', nargs='+', required=False, type=float)
    parser.add_argument('--init_memory', nargs='+', required=False, type=str)
    parser.add_argument('--distance', nargs='+', required=False, type=str)
    parser.add_argument('--a_topk', nargs='+', required=False, type=int)
    parser.add_argument('--topk_score', nargs='+', required=False, type=int)
    parser.add_argument('--loss_topk', nargs='+', required=False, type=int)

    parser.add_argument('--n_layers', nargs='+', required=False, type=int)
    parser.add_argument('--heads', nargs='+', required=False, type=int)
    parser.add_argument('--batch_size', nargs='+', required=False, type=int)
    parser.add_argument('--chunk_size', nargs='+', required=False, type=int)
    parser.add_argument('--model_name', nargs='+', required=False, type=str)
    parser.add_argument('--NMS_window', nargs='+', required=False, type=str)
    parser.add_argument('--NMS_threshold', nargs='+', required=False, type=str)
    parser.add_argument('--embedding_dim', nargs='+', required=False, type=str)
    parser.add_argument('--stride', nargs='+', required=False, type=str)
    parser.add_argument('--similar_threshold', nargs='+', required=False, type=str)
    parser.add_argument('--n_important_frames', nargs='+', required=False, type=str)
    parser.add_argument('--inference_type', nargs='+', required=False, type=str)
    parser.add_argument('--stride_4_test', nargs='+', required=False, type=str)
    parser.add_argument('--pooling_module', nargs='+', required=False, type=str)
    parser.add_argument('--use_spot_loss', nargs='+', required=False, type=str)
    parser.add_argument('--dropout_rate', nargs='+', required=False, type=str)
    parser.add_argument('--ast_t', nargs='+', required=False, type=str)
    parser.add_argument('--spot_threshold', nargs='+', required=False, type=str)
    parser.add_argument('--spot_loss_gamma', nargs='+', required=False, type=str)
    parser.add_argument('--spot_loss', nargs='+', required=False, type=str)
    args = parser.parse_args()
    # 自动选择gpu
    GPU = select_free_gpu_by_memory()

    # create main command
    main_command = f"CUDA_VISIBLE_DEVICES={GPU} python main.py {args.OA}"

    commands = []
    append_commands("lr", args.LR, commands)
    print(commands)
    append_commands("init_memory", args.init_memory, commands)
    append_commands("distance", args.distance, commands)
    append_commands("a_topk", args.a_topk, commands)
    append_commands("topk_score", args.topk_score, commands)
    append_commands("loss_topk", args.loss_topk, commands)
    print(commands)

    # append_commands("n_layers", args.n_layers, commands)
    # append_commands("heads", args.heads, commands)
    # append_commands("batch_size", args.batch_size, commands)
    # append_commands("chunk_size", args.chunk_size, commands)
    # append_commands("model_name", args.model_name, commands)
    # append_commands("embedding_dim", args.embedding_dim, commands)
    # append_commands("stride", args.stride, commands)
    # append_commands("NMS_window", args.NMS_window, commands)
    # append_commands("similar_threshold", args.similar_threshold, commands)
    # append_commands("n_important_frames", args.n_important_frames, commands)
    # append_commands("inference_type", args.inference_type, commands)
    # append_commands("stride_4_test", args.stride_4_test, commands)
    # append_commands("pooling_module", args.pooling_module, commands)
    # append_commands("use_spot_loss", args.use_spot_loss, commands)
    # append_commands("dropout_rate", args.dropout_rate, commands)
    # append_commands("NMS_threshold", args.NMS_threshold, commands)
    # append_commands("ast_t", args.ast_t, commands)
    # append_commands("spot_threshold", args.spot_threshold, commands)
    # append_commands("spot_loss", args.spot_loss, commands)
    # append_commands("spot_loss_gamma", args.spot_loss_gamma, commands)
    # commands = append_commands("NMS_threshold", args.NMS_threshold, commands, grid=True)

    if len(commands) == 0:
        commands.append(main_command)
    print("Commands : ")
    print(commands)
    for command_ in commands:
        try:
            print_red(command_)
            result = run_command(command_)
            print_red(result)
            add_record(RECORD_FILEPATH, result)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            else:
                result = run_command(command_)
                print_red(result)
                add_record(RECORD_FILEPATH, result)




#
#
# def append_commands(attr_name, values, commands):
#     global main_command
#     if values is not None and len(values) != 0:
#         # print(values)
#         # if len(values) == 1:
#         #     if len(commands) == 1:
#         #         main_command = f"{main_command} --{attr_name} {values[0]}"
#         #         commands = [main_command]
#         #     else:
#         #         for idx, command in enumerate(commands):
#         #             commands[idx] = f"{command} --{attr_name} {values[0]}"
#         #
#         # else:
#         for value in values:
#             command = f"{main_command} --{attr_name} {value}"
#             commands.append(command)
#
#     return commands
#
#
# if __name__ == '__main__':
#     import os
#
#     # GPU = 6
#     # weight_loss = 0
#     # assert weight_loss in [0, 1]
#
#     parser = ArgumentParser(description='trail for network', formatter_class=ArgumentDefaultsHelpFormatter)
#     # parser.add_argument('--GPU', required=False, type=int, default=GPU, help='ID of the GPU to use')
#     # parser.add_argument('--weight_loss', required=False, type=int, default=weight_loss, help='weight_loss')
#
#     parser.add_argument('--OA', required=False, default="", help='other arguments')
#
#     parser.add_argument('--n_layers', nargs='+', required=False, type=int)
#     parser.add_argument('--heads', nargs='+', required=False, type=int)
#     parser.add_argument('--batch_size', nargs='+', required=False, type=int)
#     parser.add_argument('--chunk_size', nargs='+', required=False, type=int)
#     parser.add_argument('--LR', nargs='+', required=False, type=float)
#     args = parser.parse_args()
#
#     # if args.weight_loss == 0:
#     #     weight_loss = "--weight_loss"
#     # else:
#     #     weight_loss = ""
#     # create main command
#     main_command = f"python main.py {args.OA}"
#
#     # mulit_arguments = ["n_layers"]
#     # print(args)
#     # print(args._get_kwargs(["n_layers"]))
#     commands = []
#     append_commands("n_layers", args.n_layers, commands)
#     append_commands("heads", args.heads, commands)
#     append_commands("batch_size", args.batch_size, commands)
#     append_commands("chunk_size", args.chunk_size, commands)
#     append_commands("lr", args.LR, commands)
#     # print(commands)
#     # agument = args._get_kwargs()
#     # argument = args._get_args()
#     # print(argument)
#     # print(agument)
#     for command in commands:
#         from print_color import print_red
#
#         try:
#             print_red(command)
#             # os.system(command)
#             result = os.popen(command).readlines()[-1]
#             print(result)
#         except Exception as e:
#             if isinstance(e, KeyboardInterrupt):
#                 raise e
#             else:
#                 print_red(e)
#                 # os.system(command)
#                 result = os.popen(command).readlines()[-1]
#                 print(result)