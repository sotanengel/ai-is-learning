# ai-is-learning (kolb-loop)

David Kolb の経験学習サイクルを既存OSS LLMへ非侵襲でアタッチする自律学習基盤。

## 開発環境セットアップ

```bash
# Takumi Guard token をセット（サプライチェーン攻撃防止）
export UV_INDEX_URL="https://<YOUR_TOKEN>@pypi.flatt.tech/simple/"

# 依存関係インストール
uv sync --extra dev

# pre-commit フック設定
uv run pre-commit install
```

## テスト実行

```bash
uv run pytest tests/unit/ -v        # ユニットテストのみ
uv run pytest tests/integration/ -v # 統合テスト（Ollamaが必要）
uv run pytest -v                    # 全テスト
```

## サーバー起動

```bash
# .env.example をコピーして設定
cp .env.example .env

# 起動
uv run kolb-loop
# → http://localhost:8080/v1/chat/completions
```

## PR マージポリシー

- 各PR: CI（ruff + mypy + pytest）全通過後にマージ
- TDD: テストコードを先に書いてから実装する
- 段階的: PR #1 → #2 → ... の順でマージ

## ブランチ戦略

| ブランチ | 内容 |
|---|---|
| `feat/foundation` | PR#1: プロジェクト基盤 |
| `feat/data-models` | PR#2: データモデル・ストレージ |
| `feat/v2-ingress-logger` | PR#3: プロキシ・ロガー |
| `feat/v2-reflection` | PR#4: 内省エンジン |
| `feat/v2-concept-injection` | PR#5: 概念蒸留・注入 |
| `feat/v2-evaluator-mcp` | PR#6: 評価器・MCP |
| `feat/v3-curator` | PR#7: 学習データ化 |
| `feat/v3-trainer` | PR#8: 学習オーケストレーター |
| `feat/v3-eval-promotion` | PR#9: 評価・プロモーション |
| `feat/observability-docker` | PR#10: 観測・Docker |

## Takumi Guard（サプライチェーン攻撃防止）

- pip/uv の index-url を `https://pypi.flatt.tech/simple/` に設定済み
- CI では GitHub Secrets `TAKUMI_GUARD_TOKEN` を使用
- 詳細: https://flatt.tech/takumi/features/guard
