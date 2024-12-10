from ultralytics import YOLOWorld, YOLO 
model = YOLOWorld('/home/zyw/data/china_tower/CV_server/YOLO-World/weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth')
prompt = 'smoke.fire.person.tree.fishing rod.big ship.small rowboat.river.chimney.farmland.factory.forest.mountain.vehicle.road.mudflow.landslide.rockfall.building. '
texts = tuple(prompt.split('.'))
model.set_classes(texts)