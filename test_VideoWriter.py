import cv2

cap = cv2.VideoCapture(0)

# 创建VideoWriter类对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('data/outVideo1.mp4', fourcc, fps, size)
# 读取视频流
while cap.isOpened():
    ret, frame = cap.read()  # 获取一帧图像
    if ret:
        frame = cv2.flip(frame, 1)  # 调整方向，可不写
        out.write(frame)  # 写入视频对象
        # 显示读取视频
        cv2.imshow('frame', frame)
        # q键关闭
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 关闭流
cap.release()
out.release()
cv2.destroyAllWindows()