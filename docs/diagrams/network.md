# Сетевая схема: прохождение пакетов

Как трафик проходит между пользователем (Telegram), хостом, виртуалками, Docker/Kubernetes и ботом.

## Контекст: хост, VM, кластер

```mermaid
flowchart TB
    subgraph Internet["Интернет"]
        User[Пользователь / Telegram клиент]
        TG[Telegram API<br/>api.telegram.org]
    end

    subgraph Host["Хост (Windows/Linux)"]
        HostNote[Машина разработки или сервер]
    end

    subgraph VM["Виртуальные машины (VirtualBox)"]
        Master[kube-master<br/>192.168.10.10]
        Node1[kube-node-1<br/>192.168.10.11]
        Node2[kube-node-2<br/>192.168.10.12]
    end

    subgraph Shared["Общая папка (VirtualBox Shared Folder)"]
        Data[Путь: home test shared xyliganimbot<br/>data, logs, models, config]
    end

    User <-->|HTTPS| TG
    TG <-->|HTTPS<br/>Long polling| Host
    Host <-.->|Shared Folder| VM
    Master <-.-> Node1
    Master <-.-> Node2
    Data <-.->|смонтирована| Master
    Data <-.->|смонтирована| Node1
    Data <-.->|смонтирована| Node2
```

## Трафик при запуске бота в Kubernetes

```mermaid
sequenceDiagram
    participant TG as Telegram API
    participant Net as Сеть узла (NAT/шлюз)
    participant Node as kube-node (pod на ноде)
    participant Bot as Под xyliganimbot
    participant PV as PersistentVolume<br/>(hostPath = общая папка)

    Note over Bot,PV: Исходящие запросы бота к Telegram
    Bot->>Net: HTTPS GET updates (long polling)
    Net->>TG: Запрос
    TG->>Net: Ответ (обновления)
    Net->>Bot: Ответ

    Note over Bot,PV: Запись логов и чтение данных
    Bot->>PV: Чтение data/, models/, config
    Bot->>PV: Запись logs/
```

## Сети в Kubernetes (Flannel)

```mermaid
flowchart LR
    subgraph PodNet["Pod network 10.10.0.0/16 (Flannel)"]
        Pod1[Pod xyliganimbot<br/>10.10.x.x]
    end
    subgraph NodeNet["Сеть узлов 192.168.10.0/24"]
        Master[master 192.168.10.10]
        N1[node-1 192.168.10.11]
        N2[node-2 192.168.10.12]
    end
    subgraph HostNet["Хост / внешний доступ"]
        GW[Шлюз 10.0.2.2]
    end

    Pod1 -->|cni0 / flannel.1| N2
    N2 <--> Master
    N1 <--> Master
    Master <--> GW
    GW <-->|Интернет| TG[Telegram API]
```

- **Узлы** общаются по 192.168.10.x (enp0s8 в примере).
- **Поды** получают IP из подсети 10.10.0.0/16 (CNI Flannel).
- Исходящий трафик из пода к Telegram идёт через сеть узла и шлюз хоста.

## Docker (один контейнер на хосте)

```mermaid
flowchart LR
    Host[Хост] --> Docker[Docker daemon]
    Docker --> Container[Контейнер xyliganimbot]
    Host --> Vol[Volumes: data, logs, models, config.yaml]
    Container --> Vol
    Container -->|HTTPS| Internet[Telegram API]
```

Трафик: контейнер → сеть хоста (bridge/NAT) → интернет → Telegram API.
