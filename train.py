import os
import time
import datetime

import torch
import logging
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import get_parse, setup_seed
from load_data import load_data, load_test_data, get_chunks

#logger
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
# 第二步，创建一个handler，用于写入日志文件
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_name = './log/train_TSMMG.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)

#加速器
# accelerator = Accelerator(gradient_accumulation_steps=2)
accelerator = Accelerator()
###################

args = get_parse()
# device = args.device
device = accelerator.device
setup_seed(args.seed)
# ---------------------------------  加载 tokenizer  --------------------------------------
# Load the GPT tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', 
                    bos_token='<|startoftext|>', 
                        eos_token='<|endoftext|>', 
                            pad_token='<|pad|>',
                                special_tokens=["<|startofsmiles|>"]) #gpt2-medium

# 获得测试数据
test_data = load_test_data(args, tokenizer)
test_dataloader = DataLoader(test_data, sampler = SequentialSampler(test_data), batch_size = args.batch_size, shuffle=False)
# ---------------------------------  加载配置文件  --------------------------------------
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
# ---------------------------------  加载模型  --------------------------------------
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
# ---------------------------------  训练参数  --------------------------------------
epochs = args.epochs
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

batch_num = int(args.sample_size /args.gpu_num / args.batch_size) + 1           # 对于每一个epoch，有这么多batch
total_steps = batch_num * epochs                                                # [number of batches] x [number of epochs]

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


model, optimizer, test_dataloader = accelerator.prepare(model, optimizer, test_dataloader)
# model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

total_t0 = time.time()
training_stats = []

for epoch_i in range(0, epochs):

    logger.info("")
    logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    logger.info('Training...')

    remaining_epoch = args.epochs - epoch_i - 1
    # ========================================
    #               Training
    # ========================================
    total_train_loss = 0
    model.train()

    total_size = 0

    # ---------------------------------  读取数据  --------------------------------------
    # 获得训练数据句柄
    chunks = get_chunks()
    passed_batch = 0
    logger.info('Number of  chunks =====> {}'.format(len(chunks)))

    for chunk in chunks:                                                        # 循环读取所有文件句柄
        accelerator.free_memory()
        while True:                                                             # 循环读取chunk内所有数据
            if total_size >= args.sample_size:                                  # 中止条件：已使用的训练数据大小大于指定训练数据大小
                break

            train_data, fetched_size = load_data(args, tokenizer, chunk)
            total_size = total_size + fetched_size

            if train_data is None:                                              # 中止条件：当前chunk无数据，进入下一个chunk
                break

            # 开始当前chunk训练
            logger.info(' Number of trained data ========> {}'.format(total_size))
            t0 = time.time()  # 计算每个chunk需要的时间

            train_dataloader = DataLoader(train_data, sampler = RandomSampler(train_data), batch_size = args.batch_size, shuffle=False)
            train_dataloader = accelerator.prepare(train_dataloader)

            for step, batch in enumerate(train_dataloader):

                b_input_ids = batch[0].to(device)
                b_labels = batch[0].to(device)
                b_masks = batch[1].to(device)

                model.zero_grad()        

                outputs = model(  b_input_ids,
                                labels=b_labels, 
                                attention_mask = b_masks,
                                token_type_ids=None
                                )

                loss = outputs[0]  

                batch_loss = loss.item()
                total_train_loss += batch_loss

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                passed_batch = passed_batch + 1

            # Calculate the average loss over all of the batches.
            # avg_train_loss = total_train_loss / batch_num 
            avg_train_loss = total_train_loss / passed_batch 
            
            # Measure how long this epoch took.
            time_elapse = time.time() - t0
            time_for_each_batch = time_elapse/step
            training_time = format_time(time_elapse)

            # remaining batches and time
            remaining_batches = batch_num - passed_batch                                # 当前epoch剩余的batch数
            remaining_time = format_time(remaining_batches * time_for_each_batch)       # 当前epoch剩余的时间

            total_remaining_time = format_time(remaining_batches * time_for_each_batch * remaining_epoch)

            if accelerator.is_main_process:
                logger.info("\n")
                logger.info(f"Current at the {epoch_i}-th epoch.")
                logger.info(" Average training loss: {0:.2f}".format(avg_train_loss))
                logger.info(" Training epoch took: {:}".format(training_time))
                logger.info(" Totally {:} needed.".format(remaining_time))
            
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            # 'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            # 'Validation Time': validation_time
        }
    )

    # ========================================
    #               Saving
    # ========================================
    if epoch_i % 5 == 4:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            output_dir = f'./model_save_{args.sample_size}_{epoch_i+1}/'
            logger.info("Saving model to %s" % output_dir)

            # Create output directory if needed
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))

            # model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            # model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("Saving success!")

logger.info("\n")
logger.info("Training complete!")
logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))




