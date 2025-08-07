import textwrap

import matplotlib
import matplotlib.pyplot as plt
import pandas
import torch
from sentencepiece import SentencePieceProcessor
from sklearn.decomposition import PCA
from torch.nn import Module, Embedding, Linear, MSELoss, LayerNorm
from torch.nn.functional import gelu
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from core.converter import lab_to_all, lab_to_hex

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 1000
LEARNING_RATE = 6e-4
SENTENCEPIECE_MODEL_PATH = 'color_unigram.model'
COLOR_WORDS_PATH = 'color_words.csv'


class ColorDataset(Dataset):
    def __init__(self, model_path, csv_path):
        self.sp = SentencePieceProcessor()
        self.sp.Load(model_path)

        with open(csv_path, 'r', encoding='utf-8') as csv:
            data = pandas.read_csv(csv)

        self.samples = []
        for _, (name, L, a, b) in data.iterrows():
            tokens = self.sp.Encode(name, out_type=int)
            self.samples.append((tokens, [L, a, b]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, lab = self.samples[idx]
        return torch.tensor(tokens), torch.tensor(lab, dtype=torch.float32)


def collate_fn(batch):
    tokens, labs = zip(*batch)
    tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
    labs_tensor = torch.stack(labs)
    return tokens_padded.to(DEVICE), labs_tensor.to(DEVICE)


class ColorMLP(Module):
    def __init__(self, vocab_size):
        super().__init__()

        # 1️⃣ 词嵌入层：将 token_id 映射为 128维 embedding 向量
        self.embedding = Embedding(vocab_size, 128, device=DEVICE)

        # 2️⃣ 第一线性层：128 → 256，扩大维度用于表达能力增强
        self.linear1 = Linear(128, 256, device=DEVICE)
        self.norm1 = LayerNorm(256)

        # 3️⃣ 第二线性层：256 → 256，形成隐藏空间的非线性组合
        self.linear2 = Linear(256, 256, device=DEVICE)
        self.norm2 = LayerNorm(256)

        # 4️⃣ 第三线性层：256 → 128，重新压缩回 embedding 维度
        self.linear3 = Linear(256, 128, device=DEVICE)
        self.norm3 = LayerNorm(128)

        # 5️⃣ 输出层：128 → 3，对应 CIELAB 中的 L, a, b 三个分量
        self.linear4 = Linear(128, 3, device=DEVICE)

    def forward(self, input_ids):
        # input_ids: [B, T]，B 是 batch_size，T 是 token 数（通常 < 5）

        # 🧩 第一步：获取词嵌入向量
        x = self.embedding(input_ids)  # [B, T, 128]

        # 🧠 第二步：对 token embedding 求平均，得到句子表示（颜色整体语义）
        x = x.mean(dim=1)  # [B, 128]

        # ⚙️ 第三步：MLP 三层非线性变换
        x = gelu(self.norm1(self.linear1(x)))  # [B, 256]
        x = gelu(self.norm2(self.linear2(x)))  # [B, 256]
        x = gelu(self.norm3(self.linear3(x)))  # [B, 128]

        # 🎯 第四步：输出层，预测归一化后的 Lab 值
        return self.linear4(x)  # [B, 3]


def train():
    sp = SentencePieceProcessor()
    sp.Load(SENTENCEPIECE_MODEL_PATH)
    vocab_size = sp.GetPieceSize()

    dataset = ColorDataset(SENTENCEPIECE_MODEL_PATH, COLOR_WORDS_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    model = ColorMLP(vocab_size).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
    loss_fn = MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count

        # ✅ CosineAnnealing 每轮都 step
        cosine_scheduler.step()

        # ✅ ReduceLROnPlateau 监控 loss，自动衰减学习率（但不 step）
        plateau_scheduler.step(avg_loss)

        # ✅ 可选打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}, LR: {current_lr:.15f}')

    torch.save(model.state_dict(), 'color_model.pt')
    print('✅ 训练完成，模型已保存')


class ColorSemanticMapper:
    def __init__(self):
        self.sp = SentencePieceProcessor()
        self.sp.Load(SENTENCEPIECE_MODEL_PATH)
        vocab_size = self.sp.GetPieceSize()

        self.model = ColorMLP(vocab_size).to(DEVICE)
        self.model.load_state_dict(torch.load('color_model.pt'))
        self.model.eval()

    def __call__(self, word: str) -> str:
        """
        依据颜色名称预测颜色值
        :param word: 颜色名称
        :return: 元组，[rgb, hex, hsv, hsl, lab]
        """
        tokens = torch.tensor(self.sp.Encode(word, out_type=int)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            lab = self.model(tokens).squeeze().cpu().numpy()
        rgb, _hex, hsv, hsl, lab_int = lab_to_all(*lab)
        return textwrap.dedent(f"""\
        颜色: {word}
        --------------------
        RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}
        HEX: {_hex}
        HSV: {hsv[0]}, {hsv[1]}, {hsv[2]}
        HSL: {hsl[0]}, {hsl[1]}, {hsl[2]}
        LAB: {lab_int[0]}, {lab_int[1]}, {lab_int[2]}\
        """)


def predict(word: str, sp: SentencePieceProcessor, model: ColorMLP) -> tuple[float, float, float]:
    tokens = torch.tensor(sp.Encode(word, out_type=int)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        lab = model(tokens).squeeze().cpu().numpy()
    return lab


def model_pca():
    matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

    color = ColorSemanticMapper()

    sp = SentencePieceProcessor()
    sp.Load(SENTENCEPIECE_MODEL_PATH)
    vocab_size = sp.GetPieceSize()

    model = ColorMLP(vocab_size).to(DEVICE)
    model.load_state_dict(torch.load('color_model.pt'))
    model.eval()

    # ✅ 加载颜色词表（用于预测）
    df = pandas.read_csv(COLOR_WORDS_PATH)
    color_names = df['name'].tolist()

    # ✅ 获取所有颜色词的预测值
    predicted_lab = []
    for word in color_names:
        lab, _, _, _, _, _ = color(word)
        predicted_lab.append(lab)

    # ✅ PCA 投影
    pca = PCA(n_components=2)
    lab_2d = pca.fit_transform(predicted_lab)

    # ✅ 可视化
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(lab_2d):
        plt.scatter(x, y, color=lab_to_hex(*predicted_lab[i]), s=100, edgecolors='k')

    plt.title("颜色词预测嵌入的 PCA 分布图（模型输出）")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # train()
    # model_pca()
    color_mapper = ColorSemanticMapper()
    print(color_mapper('红'))
