# Развертывание через SSH

Инструкция предполагает, что вы подключаетесь по SSH к хосту или к узлам Kubernetes и выполняете команды из корня проекта (например, `/home/test/shared/xyliganimbot`).

## Подключение

```bash
# К мастер-ноде Kubernetes (или к единственной VM с Docker)
ssh user@192.168.10.10

# Переход в каталог проекта (если общая папка смонтирована здесь)
cd /home/test/shared/xyliganimbot
```

Дальнейшие шаги совпадают с разделами **«Запуск в Docker»** и **«Запуск в Kubernetes»** в `README.md`.

---

## Docker (один контейнер на хосте)

После входа по SSH:

1. Подготовка: `config.yaml`, `.env` (см. README, раздел «Запуск в Docker»).
2. Сборка: `docker build -t xyliganimbot .`
3. Запуск: команда `docker run -d ...` с томами и `--env-file` из README.
4. Логи: `docker logs -f xyliganimbot`
5. Остановка: `docker stop xyliganimbot`, `docker rm xyliganimbot`

---

## Kubernetes (кластер из нескольких VM)

### 1. Подключение к мастер-ноде

```bash
ssh user@192.168.10.10
cd /home/test/shared/xyliganimbot
```

### 2. Подготовка образа

Если образ собирается на этом же хосте:

```bash
docker build -t xyliganimbot:latest .
docker save xyliganimbot:latest -o xyliganimbot-latest.tar
```

Если общая папка доступна на всех нодах, файл `xyliganimbot-latest.tar` уже в ней.

### 3. Импорт образа на каждую ноду

Подключитесь по SSH к **каждой** ноде (master, node-1, node-2) и выполните:

```bash
cd /home/test/shared/xyliganimbot
sudo ctr -n k8s.io images import xyliganimbot-latest.tar
sudo ctr -n k8s.io images list | grep xyliganimbot
```

Или одной командой с вашей машины (если настроен SSH по ключам):

```bash
for node in 192.168.10.10 192.168.10.11 192.168.10.12; do
  ssh user@$node "cd /home/test/shared/xyliganimbot && sudo ctr -n k8s.io images import xyliganimbot-latest.tar"
done
```

### 4. Применение манифестов (на мастер-ноде)

```bash
ssh user@192.168.10.10
cd /home/test/shared/xyliganimbot

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pv.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
kubectl create secret generic xyliganimbot-secrets --namespace=xyliganimbot --from-env-file=.env
kubectl apply -f k8s/deployment.yaml
```

### 5. Проверка

```bash
kubectl get pods -n xyliganimbot
kubectl logs -f deployment/xyliganimbot -n xyliganimbot
```

Пути в `k8s/pv.yaml` должны соответствовать путям на нодах (например, `/home/test/shared/xyliganimbot/data`, `logs`, `models`).
