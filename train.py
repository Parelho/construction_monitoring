from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# Caso esteja recebendo um erro de memória, tente diminuir a quantidade de batch para não ultrapassar o limite de VRAM da sua GPU
# Caso esteja travando e/ou fechando o terminal, tente diminuir o número de workers para reduzir a carga na memória RAM
model.train(
   data=".\construction\data.yaml",
   epochs=100,
)

metrics = model.val()
print(metrics)