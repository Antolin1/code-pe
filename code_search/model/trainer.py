import logging

import evaluate
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torchmetrics.functional import retrieval_reciprocal_rank
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger()


def train(train_set, valid_set, model, checkpoint, batch_size_train=16, lr=5e-5, epochs=30, patience=5,
          gradient_accumulation=1, max_grad_norm=1, wandb_enabled=False, batch_size_eval=64,
          log_steps=100):
    train_set.set_format("torch")
    valid_set.set_format("torch")
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size_train)
    eval_dataloader = DataLoader(valid_set, batch_size=batch_size_eval, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, eps=1e-8)

    criterion = torch.nn.CrossEntropyLoss()

    logger.info('Training phase!')
    logger.info(f'Effective batch size: {batch_size_train * gradient_accumulation}')
    logger.info(f'Initial lr: {lr}')
    logger.info(f'Epochs: {epochs}')
    logger.info(f'Parameters: {sum(map(torch.numel, filter(lambda p: p.requires_grad, model.parameters())))}')

    best_mrr = -float('inf')
    patience_count = 0
    num_training_steps = epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps * 0.1,
                                                num_training_steps=num_training_steps)
    progress_bar = tqdm(range(num_training_steps))
    steps = 0
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        model.train()
        for j, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            emb_code, emb_nl = model(input_ids_code=batch['input_ids_code'], inputs_ids_nl=batch['input_ids_nl'],
                                     attention_mask_code=batch['attention_mask_code'],
                                     attention_mask_nl=batch['attention_mask_nl'])
            scores = torch.matmul(emb_nl, torch.transpose(emb_code, 0, 1))
            loss = criterion(scores, torch.arange(scores.shape[0]).to(device))

            train_loss += loss.item()
            loss = loss / gradient_accumulation

            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)

            if ((j + 1) % gradient_accumulation == 0) or (j + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            progress_bar.update(1)
            steps += 1
            if steps % log_steps == 0:
                logger.info(
                    f'Epoch {epoch} | step={steps} | train_loss={train_loss / (j + 1):.4f}'
                )
                if wandb_enabled:
                    wandb.log({'train/loss': train_loss / (j + 1),
                               'step': steps,
                               'epoch': epoch})

        # evaluate
        accuracy_eval, mrr, eval_loss = evaluation(model, eval_dataloader, device, criterion)

        # save best model
        if mrr > best_mrr:
            logger.info('Saving model!')
            torch.save(model.state_dict(), checkpoint)
            logger.info(f'Model saved: {checkpoint} Best mrr {mrr:.4f}')
            patience_count = 0
            best_mrr = mrr
            if wandb_enabled:
                wandb.run.summary["best_mrr"] = best_mrr
        else:
            patience_count += 1
        if patience_count == patience:
            logger.info('Stopping training loop (out of patience).')
            break

        logger.info(
            f'Epoch {epoch} | train_loss={train_loss / len(train_dataloader):.4f} '
            f'| eval_mrr={mrr:.4f} | eval_acc={accuracy_eval:.4f} | eval_loss={eval_loss}'
        )
        if wandb_enabled:
            wandb.log({'train/loss': train_loss / len(train_dataloader),
                       'step': steps,
                       'epoch': epoch,
                       'val/mrr': mrr,
                       'val/acc': accuracy_eval,
                       'val/loss': eval_loss})


def evaluation(model, dataloader, device, criterion=torch.nn.CrossEntropyLoss()):
    model.eval()
    metric = evaluate.load("accuracy")
    rrs = []
    eval_loss = 0.0
    for batch in tqdm(dataloader, desc='Evaluation loop'):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            emb_code, emb_nl = model(input_ids_code=batch['input_ids_code'], inputs_ids_nl=batch['input_ids_nl'],
                                     attention_mask_code=batch['attention_mask_code'],
                                     attention_mask_nl=batch['attention_mask_nl'])
            scores = torch.matmul(emb_nl, torch.transpose(emb_code, 0, 1))
            predictions = torch.argmax(scores, dim=-1)
            loss = criterion(scores, torch.arange(scores.shape[0]).to(device))
            eval_loss += loss.item()
            # accuracy
            metric.add_batch(predictions=predictions, references=torch.arange(scores.shape[0]).to(device))
            # mrr, I think that this is incorrect, check it
            for scs, tgt in zip(scores, torch.eye(scores.shape[0]).to(device)):
                rr = retrieval_reciprocal_rank(scs, tgt).item()
                rrs.append(rr)

    accuracy_eval = metric.compute()['accuracy']
    return accuracy_eval, np.mean(rrs), eval_loss/len(dataloader)


def eval_test(test_set, model, batch_size=64):
    test_set.set_format("torch")
    test_dataloader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    accuracy_test, mrr, _ = evaluation(model, test_dataloader, device)
    logger.info(f'Test accuracy: {accuracy_test:.4f}')
    logger.info(f'Test MRR: {mrr:.4f}')
