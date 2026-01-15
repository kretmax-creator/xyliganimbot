# Диаграмма потоков данных - Итерация 3

Диаграмма показывает потоки данных в системе на этапе итерации 3 (модуль конфигурации).

## Поток данных при запуске бота

```mermaid
flowchart TD
    Start([Запуск бота]) --> LoadConfig[load_config<br/>config.yaml]
    LoadConfig --> LoadSecrets[load_secrets<br/>.env файл]
    LoadSecrets --> Validate[validate_config<br/>Проверка обязательных параметров]
    
    Validate -->|Ошибка| ErrorExit[Ошибка<br/>Выход из программы]
    Validate -->|Успех| GetWhitelists[get_whitelists<br/>Извлечение белых списков]
    
    GetWhitelists --> GetLoggingConfig[get_logging_config<br/>Настройки логирования]
    GetLoggingConfig --> SetupLogging[setup_logging<br/>Инициализация логирования]
    
    SetupLogging --> CreateApp[create_application<br/>Создание Telegram приложения]
    CreateApp --> RunPolling[run_polling<br/>Запуск long polling]
    
    RunPolling --> WaitMessages[Ожидание сообщений]
    
    style ErrorExit fill:#ffcccc
    style Start fill:#ccffcc
    style WaitMessages fill:#ccccff
```

## Поток данных при обработке сообщения

```mermaid
flowchart TD
    TelegramAPI[Telegram API<br/>Входящее сообщение] --> ReceiveUpdate[Получение Update]
    
    ReceiveUpdate --> CheckUser{Проверка<br/>user/chat/message}
    CheckUser -->|Отсутствует| LogWarning[Логирование<br/>предупреждения]
    CheckUser -->|Присутствует| CheckWhitelist[is_user_allowed<br/>Проверка белого списка]
    
    CheckWhitelist -->|Доступ запрещен| LogDenied[Логирование<br/>доступ запрещен]
    CheckWhitelist -->|Доступ разрешен| LogMessage[Логирование<br/>входящего сообщения]
    
    LogMessage --> AuditLog[Запись в<br/>audit.log]
    
    LogWarning --> End([Конец обработки])
    LogDenied --> End
    AuditLog --> End
    
    style TelegramAPI fill:#ccccff
    style CheckWhitelist fill:#ffffcc
    style LogMessage fill:#ccffcc
    style End fill:#ffcccc
```

## Структура данных конфигурации

```mermaid
flowchart LR
    ConfigYAML[config.yaml] --> ConfigDict[Словарь конфигурации]
    EnvFile[.env файл] --> SecretsDict[Словарь секретов]
    
    ConfigDict --> Whitelists[Белые списки<br/>users, chats, admins]
    ConfigDict --> LoggingConfig[Настройки логирования<br/>level, file, audit_file, etc.]
    ConfigDict --> GoogleDocsConfig[Настройки Google Docs<br/>url, document_id]
    ConfigDict --> CacheConfig[Настройки кэша<br/>update_interval_days, auto_update]
    
    SecretsDict --> TelegramToken[TELEGRAM_BOT_TOKEN]
    SecretsDict --> GoogleKey[GOOGLE_SERVICE_ACCOUNT_KEY]
    SecretsDict --> LogLevel[LOG_LEVEL]
    
    style ConfigYAML fill:#ccffcc
    style EnvFile fill:#ffcccc
    style Whitelists fill:#ffffcc
```

## Компоненты системы

```mermaid
graph TB
    subgraph "Модуль конфигурации (src/config.py)"
        LoadConfig[load_config]
        LoadSecrets[load_secrets]
        Validate[validate_config]
        GetWhitelists[get_whitelists]
        GetLoggingConfig[get_logging_config]
    end
    
    subgraph "Модуль бота (src/bot.py)"
        Main[main]
        IsUserAllowed[is_user_allowed]
        IsAdmin[is_admin]
        HandleMessage[handle_message]
        CreateApp[create_application]
    end
    
    subgraph "Модуль логирования (src/logging.py)"
        SetupLogging[setup_logging]
        GetLogger[get_logger]
        GetAuditLogger[get_audit_logger]
    end
    
    subgraph "Внешние источники"
        ConfigFile[config.yaml]
        EnvFile[.env]
        TelegramAPI[Telegram API]
    end
    
    ConfigFile --> LoadConfig
    EnvFile --> LoadSecrets
    LoadConfig --> Validate
    LoadSecrets --> Validate
    Validate --> GetWhitelists
    Validate --> GetLoggingConfig
    
    GetLoggingConfig --> SetupLogging
    SetupLogging --> GetLogger
    SetupLogging --> GetAuditLogger
    
    GetWhitelists --> IsUserAllowed
    GetWhitelists --> IsAdmin
    
    Main --> LoadConfig
    Main --> LoadSecrets
    Main --> Validate
    Main --> GetWhitelists
    Main --> GetLoggingConfig
    Main --> SetupLogging
    Main --> CreateApp
    
    CreateApp --> HandleMessage
    TelegramAPI --> HandleMessage
    HandleMessage --> IsUserAllowed
    HandleMessage --> GetLogger
    HandleMessage --> GetAuditLogger
    
    style LoadConfig fill:#ccffcc
    style LoadSecrets fill:#ccffcc
    style Validate fill:#ffffcc
    style GetWhitelists fill:#ccccff
    style HandleMessage fill:#ffcccc
```

## Последовательность операций при старте

```mermaid
sequenceDiagram
    participant Main as main()
    participant Config as config.py
    participant Secrets as .env
    participant Validate as validate_config()
    participant Logging as logging.py
    participant Bot as Telegram Bot
    
    Main->>Config: load_config()
    Config-->>Main: config dict
    
    Main->>Secrets: load_secrets()
    Secrets-->>Main: secrets dict
    
    Main->>Validate: validate_config(config, secrets)
    Validate->>Validate: Проверка TELEGRAM_BOT_TOKEN
    Validate->>Validate: Проверка белых списков
    alt Ошибка валидации
        Validate-->>Main: ValueError
        Main->>Main: sys.exit(1)
    else Успех
        Validate-->>Main: OK
    end
    
    Main->>Config: get_whitelists(config)
    Config-->>Main: whitelists dict
    
    Main->>Config: get_logging_config(config)
    Config-->>Main: logging_config dict
    
    Main->>Logging: setup_logging(...)
    Logging-->>Main: OK
    
    Main->>Bot: create_application(token)
    Bot-->>Main: Application
    
    Main->>Bot: run_polling()
    Bot->>Bot: Ожидание сообщений
```

## Последовательность обработки сообщения

```mermaid
sequenceDiagram
    participant Telegram as Telegram API
    participant Bot as bot.py
    participant Config as config.py
    participant Logger as logging.py
    participant Audit as audit.log
    
    Telegram->>Bot: Update (сообщение)
    Bot->>Bot: handle_message(update)
    
    Bot->>Bot: Проверка user/chat/message
    alt Отсутствует user/chat/message
        Bot->>Logger: warning()
        Bot->>Bot: return
    end
    
    Bot->>Config: is_user_allowed(user_id, chat_id)
    Config->>Config: Проверка whitelists
    alt Доступ запрещен
        Config-->>Bot: False
        Bot->>Logger: warning(access denied)
        Bot->>Bot: return
    else Доступ разрешен
        Config-->>Bot: True
    end
    
    Bot->>Logger: info(Received message)
    Bot->>Audit: info(Message from user)
    
    Bot->>Bot: Конец обработки
```
