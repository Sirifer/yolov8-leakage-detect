from ultralytics import YOLO

def main():
    # model = YOLO('/cms/user/huangsuyun/dataset/runs/afterbonding/train_1_105_1000_aug_alb/weights/best.pt')
    model = YOLO('/cms/user/huangsuyun/dataset/runs/afterbonding/train_1_105_1000/weights/best.pt')
    model.val(data= 'test.yaml',
    conf=0.05,
    iou=0.5     )    # 更低的置信度阈值)

if __name__ == "__main__":
    main()

