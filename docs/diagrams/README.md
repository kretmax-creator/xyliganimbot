# Диаграммы проекта (Mermaid)

В папке `docs/diagrams/` хранятся диаграммы в формате Mermaid (в .md файлах). Их можно просматривать в GitHub, в IDE с поддержкой Mermaid или на [mermaid.live](https://mermaid.live).

## Уже есть

- **data_flow_iteration3.md** — потоки данных при запуске и обработке сообщений (конфиг, белые списки).
- **data_flow_semantic_search.md** — потоки при запуске, импорте документа и семантическом поиске.

---

## Диаграммы (итерация 15)

| № | Диаграмма | Файл |
|---|-----------|------|
| 1 | **Архитектура (компоненты)** — три слоя: Telegram-адаптер → доменная логика → доступ к данным | `architecture.md` |
| 2 | **Маршрутизация команд** — входящее сообщение → whitelist → /help, /search, /admin или поиск | `command_routing.md` |
| 4 | **Развертывание** — сборка образа → save → import на нодах → apply манифестов | `deployment.md` |
| 7 | **Сетевая схема** — прохождение пакетов: хост, виртуалки, Docker/Kubernetes, Telegram API | `network.md` |
| 8 | **Ревью существующих** — `data_flow_iteration3.md`, `data_flow_semantic_search.md` | — |
