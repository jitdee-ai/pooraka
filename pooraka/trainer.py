import logging
from pooraka.utils import AverageMeter, accuracy
import torch.nn as nn
import torch
import time
import numpy as np
import random

def train_cifar(args, train_queue, model, criterion, optimizer, logging_mode=False):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        
        optimizer.zero_grad()

        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        
        if logging_mode:
            if step % args.report_freq == 0:
                logging.info('train %03d %f %f %f mem %.2f MB', step, objs.avg, top1.avg, top5.avg, round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

    return top1.avg, objs.avg


def train_imagenet(args, train_queue, model, criterion, optimizer, logging_mode=False):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if logging_mode:
            if step % args.report_freq == 0:
                end_time = time.time()
                if step == 0:
                    duration = 0
                    start_time = time.time()
                else:
                    duration = end_time - start_time
                    start_time = time.time()
                logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs', 
                                        step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)

    return top1.avg, objs.avg