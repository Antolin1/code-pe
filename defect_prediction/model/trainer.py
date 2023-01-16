import logging

import torch.cuda
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate


def train(train_set, valid_set, model, checkpoint, batch_size=32, lr=1e-4, epochs=30, patience=5):
    logger = logging.getLogger()
    train_set.set_format("torch")
    valid_set.set_format("torch")
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(valid_set, batch_size=batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_acc = -float('inf')
    patience_count = 0
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = 0.0
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logit = model(x=batch['input_ids'], mask=batch['attention_mask'], label=batch['target'])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
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


def eval_test(test_set, model, batch_size=32):
    logger = logging.getLogger()
    test_set.set_format("torch")
    test_dataloader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    accuracy_test = evaluation(model, test_dataloader, device)
    logger.info(f'Test accuracy: {accuracy_test:.4f}')


def evaluation(model, dataloader, device):
    model.eval()
    metric = evaluate.load("accuracy")
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logit = model(x=batch['input_ids'], mask=batch['attention_mask'])
        predictions = torch.argmax(logit, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["target"])

    accuracy_eval = metric.compute()['accuracy']
    return accuracy_eval
