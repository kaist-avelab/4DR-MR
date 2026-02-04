import cv2
import os
import re

# 이미지 파일이 저장된 디렉토리
plt_save_path = '/home/ave/song/vis_results/10/detect'
# 동영상 파일 저장 경로
video_path = '/home/ave/song/vis_results/10/detect.avi'

def extract_number(filename):
    # 정규 표현식을 사용해 파일명에서 숫자만 추출합니다.
    match = re.search(int, r'\d+', filename)
    if match:
        return int(match.group(0))
    return None

# 동영상 파일의 프레임 크기 및 프레임레이트 지정
frame_width = 1500
frame_height = 3000
# frame_width = 1920
# frame_height = 1080
fps = 10

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(0))
    return 0  # 숫자가 없는 경우 0을 반환

# VideoWriter 객체 초기화
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

# 이미지 파일 목록 불러오기
image_files = [f for f in os.listdir(plt_save_path) if f.endswith('.png')]
# image_files.sort()  # 파일명 순서대로 정렬
image_files.sort(key=extract_number)

# 이미지 파일을 순차적으로 읽어 동영상 프레임으로 추가
for image_file in image_files:
    plt_file = os.path.join(plt_save_path, image_file)
    vis_bev = cv2.imread(plt_file)
    
    # 이미지 크기 조정 (동영상 프레임 크기와 일치하도록)
    vis_bev = cv2.resize(vis_bev, (frame_width, frame_height))
    
    video.write(vis_bev)

# 동영상 저장 및 자원 해제
video.release()
