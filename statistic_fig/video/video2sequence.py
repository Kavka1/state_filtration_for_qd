from state_filtration_for_qd.utils import confirm_path_exist
import cv2


video_path = '/home/xukang/Project/state_filtration_for_qd/statistic_fig/video/walker_8.mp4'
fig_path = '/home/xukang/Project/state_filtration_for_qd/statistic_fig/video/walker_left_foot_raise_up/'
confirm_path_exist(fig_path)

vidcap = cv2.VideoCapture(video_path)
success,image = vidcap.read()
count = 0
success = True
while success:
    if count > 5 and count % 1 == 0:
        cv2.imwrite(f"{fig_path}%d.jpg"%count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, count * 10 )
    count += 1

    if count > 30:
        break
