import logging

import evaluate
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(train_set, valid_set, model, checkpoint, batch_size=32, lr=1e-4, epochs=30, patience=5):
    logger = logging.getLogger()
    train_set.set_format("torch")
    valid_set.set_format("torch")
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(valid_set, batch_size=64, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = -float('inf')
    patience_count = 0
    num_training_steps = epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    steps = 0
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            emb_code, emb_nl = model(input_ids_code=batch['input_ids_code'], inputs_ids_nl=batch['input_ids_nl'],
                                     attention_mask_code=batch['attention_mask_code'],
                                     attention_mask_nl=batch['attention_mask_nl'])
            scores = torch.matmul(emb_nl, torch.transpose(emb_code, 0, 1))
            loss = criterion(scores, torch.arange(scores.shape[0]).to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            steps += 1
            if steps % 500 == 0:
                logger.info(
                    f'Epoch {epoch} | step={steps} |train_loss={train_loss / steps:.4f}'
                ) # not /steps

        # evaluate
        accuracy_eval = evaluation(model, eval_dataloader, device)

        # save best model
        if accuracy_eval > best_acc:
            logger.info('Saving model!')
            torch.save(model.state_dict(), checkpoint)
            logger.info(f'Model saved: {checkpoint}')
            patience_count = 0
            best_acc = accuracy_eval
        else:
            patience_count += 1
        if patience_count == patience:
            logger.info('Stopping training loop (out of patience).')
            break

        logger.info(
            f'Epoch {epoch} | train_loss={train_loss / len(train_dataloader):.4f} | eval_acc={accuracy_eval:.4f}'
        )


def evaluation(model, dataloader, device):
    model.eval()
    metric = evaluate.load("accuracy")
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            emb_code, emb_nl = model(input_ids_code=batch['input_ids_code'], inputs_ids_nl=batch['input_ids_nl'],
                                     attention_mask_code=batch['attention_mask_code'],
                                     attention_mask_nl=batch['attention_mask_nl'])
            scores = torch.matmul(emb_nl, torch.transpose(emb_code, 0, 1))
            predictions = torch.argmax(scores, dim=-1)
            metric.add_batch(predictions=predictions, references=torch.arange(scores.shape[0]).to(device))

    accuracy_eval = metric.compute()['accuracy']
    return accuracy_eval
