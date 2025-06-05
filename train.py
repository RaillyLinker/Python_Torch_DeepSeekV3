import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# DeepSeek-V3 구현체가 정의된 파일을 import합니다.
# (만약 같은 파일에 정의되어 있다면 따로 import 없이 사용하셔도 됩니다.)
from model import DeepSeekV3

# ============================
# 1. 하이퍼파라미터 설정
# ============================
BATCH_SIZE = 2
SEQ_LEN = 512
EPOCHS = 12
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_MODEL = "gpt2"  # 또는 "mistralai/Mistral-7B-v0.1"

# DeepSeek-V3 전용 하이퍼파라미터 (예시)
MODEL_DIM = 512       # 모델 d_model 크기
NUM_LAYERS = 4        # 디코더 레이어 개수
NUM_HEADS = 8         # 멀티헤드 어텐션 헤드 수
HIDDEN_DIM = 2048     # FFN 내부 차원 (d_ff)
NUM_EXPERTS = 4       # MoE 전문가 수
TOP_K = 2             # MoE 상위 K개 전문가 선택
MAX_LEN = SEQ_LEN     # 최대 시퀀스 길이 (RoPE용)
DROPOUT = 0.1         # 드롭아웃 확률
NOISE_STD = 1.0       # MoE 노이즈 표준편차
CAPACITY_FACTOR = 1.0 # MoE capacity factor

# ============================
# 2. 토크나이저 준비
# ============================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
# GPT 계열 모델은 pad_token이 없으므로 eos_token을 pad_token으로 지정
tokenizer.pad_token = tokenizer.eos_token

# ============================
# 3. 데이터셋 로드 및 토크나이즈
# ============================
raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")


def tokenize(example):
    # 반환값으로 "input_ids" 키만 남도록 합니다.
    return tokenizer(example["text"], return_attention_mask=False)


tokenized_datasets = raw_datasets.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

# ============================
# 4. 전체 토큰을 이어붙여서 SEQ_LEN 단위로 슬라이싱
# ============================
class TextDataset(Dataset):
    def __init__(self, tokenized_data, seq_len):
        all_tokens = []
        for item in tokenized_data["input_ids"]:
            all_tokens.extend(item)
        total_len = len(all_tokens)

        # seq_len 단위로 끊어서 sequence 생성
        self.seq_data = [
            torch.tensor(all_tokens[i : i + seq_len], dtype=torch.long)
            for i in range(0, total_len - seq_len, seq_len)
        ]

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        x = self.seq_data[idx]         # (seq_len,)
        y = x.clone()                  # 언어 모델링: input과 target을 동일하게 사용하되 shift는 학습 루프에서 처리
        return x, y


train_dataset = TextDataset(tokenized_datasets["train"], SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================
# 5. DeepSeek-V3 모델 초기화
# ============================
model = DeepSeekV3(
    vocab_size=tokenizer.vocab_size,
    dim=MODEL_DIM,
    n_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_experts=NUM_EXPERTS,
    top_k=TOP_K,
    max_len=MAX_LEN,
    dropout=DROPOUT,
    noise_std=NOISE_STD,
    capacity_factor=CAPACITY_FACTOR,
).to(DEVICE)

# ============================
# 6. 옵티마이저 및 손실 함수
# ============================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

# ============================
# 7. 학습 루프
# ============================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
    for batch in loop:
        x, y = [b.to(DEVICE) for b in batch]  # x, y: (BATCH_SIZE, SEQ_LEN)

        optimizer.zero_grad()

        # 수정된 DeepSeek-V3: forward(x) → logits만 반환 (shape: B x T x V)
        logits = model(x)  # logits: (batch_size, seq_len, vocab_size)

        # 1) 언어 모델링용 CE 손실 (shifted)
        #    - logits[:, :-1, :] (B, T-1, V) 와 y[:, 1:] (B, T-1) 을 맞춰서 계산
        #    - reshape: (B*(T-1), V) vs (B*(T-1),)
        ce_loss = loss_fn(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            y[:, 1:].contiguous().view(-1)
        )

        # 2) auxiliary loss가 제거되었으므로, CE loss만 사용
        loss = ce_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

    # 매 에폭마다 모델을 저장 (원하는 경로/이름으로 변경 가능)
    torch.save(model.state_dict(), f"deepseekv3_epoch{epoch + 1}.pth")
