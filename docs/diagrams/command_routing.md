# Маршрутизация команд

Как входящее сообщение попадает к нужному обработчику (bot.py, handlers).

## Выбор обработчика

```mermaid
flowchart TD
    Msg[Входящее сообщение] --> CheckAccess{Чат в<br/>chats.allowed?}
    CheckAccess -->|Нет| Deny[Access denied<br/>Лог, ответа нет]
    CheckAccess -->|Да| IsCmd{Команда?}
    IsCmd -->|/help| Help[handle_help_command]
    IsCmd -->|/search ...| SearchCmd[handle_search_query<br/>аргумент = запрос]
    IsCmd -->|/search без аргументов| Hint[Подсказка:<br/>/search ваш запрос]
    IsCmd -->|/admin ...| AdminCheck{Пользователь<br/>в admins?}
    AdminCheck -->|Нет| AdminDeny[«Только администраторам»]
    AdminCheck -->|Да| Admin[handle_admin_command<br/>load_model / vectorize]
    IsCmd -->|Текст в группе с @бот| SearchMsg[handle_search_query<br/>текст без упоминания]
    IsCmd -->|Текст в личке без /search| PrivateHint[«Используйте /search запрос»]
    IsCmd -->|Иначе| Skip[Не обрабатывается]
```

## Порядок регистрации обработчиков (bot.py)

1. `CommandHandler("help", ...)` — /help  
2. `CommandHandler("admin", ...)` — /admin (после проверки admins)  
3. `MessageHandler(TEXT & GROUPS & Mention(bot))` — текст с упоминанием бота в группе  
4. `CommandHandler("search", ...)` — /search запрос (или подсказка, если запрос пустой)
