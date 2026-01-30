# Changelog

Все значимые изменения проекта документируются в этом файле.

## [Unreleased]

- Доработка фильтрации нерелевантных запросов (в бэклоге)

## [0.1.0] — 2026-01

### MVP

- **Telegram-бот:** long polling, команды `/help`, `/search`, обработка текста с упоминанием бота в группах.
- **Белые списки:** доступ по чатам (`chats.allowed`), админские команды по `admins`.
- **Поиск:** семантический поиск (embedding-модель `intfloat/multilingual-e5-small`), ранжирование по score, цитаты из разделов.
- **Админские команды:** `/admin load_model`, `/admin vectorize`.
- **Конфигурация:** `config.yaml`, секреты в `.env`.
- **Логирование и аудит:** `logs/app.log`, `logs/audit.log`.
- **Docker:** образ для запуска с томами для data, logs, models, config.
- **Kubernetes:** манифесты (namespace, PV/PVC, ConfigMap, Secret, Deployment), инструкции по развертыванию и по SSH.
- **Документация:** README, docs/commands.md, docs/deploy_ssh.md, диаграммы (Mermaid), CHANGELOG.

Импорт контента из Google Docs выполняется отдельным скриптом вне бота.
