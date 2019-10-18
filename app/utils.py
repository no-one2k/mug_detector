import numpy as np
import os
import cv2
import json

YOLO_DETECTOR = 'yolo'
BLUE_DETECTOR = 'blue'
DEFAULT_DETECTOR = BLUE_DETECTOR


def get_all_frames(fpath):
    cap = cv2.VideoCapture(fpath)
    res = []
    if not cap.isOpened(): 
        print("Error opening video stream or file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            res.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else: 
            break
    cap.release()
    return res

def save_video(frames, out_fpath, dsize=(300, 200)):
    import imageio
    print(out_fpath)
    with imageio.get_writer(out_fpath, fps=10) as writer:
        for f in frames:
            resized = cv2.resize(f, dsize=dsize)
            writer.append_data(resized)
    
def save_results(input_fpath, res_frames, res_boxes):
    out_dir = get_out_dir(input_fpath)
    print(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, fr in enumerate(res_frames):
        resized = cv2.resize(fr, dsize=(300, 200))
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f'sec_{i:05d}.jpg'), resized)  
        
    boxes_js = {f'sec_{i:05d}.jpg': [] if box is None else [box] for i, box in enumerate(res_boxes)}
    with open(os.path.join(out_dir, 'boxes.json'), "wt") as f:
        json.dump(boxes_js, f)

def select_by_color(frame, colors, eps=20):
    mask = np.zeros_like(frame[:,:,0], dtype=np.uint8)
    for color in colors:
        lower = np.clip(color - eps, 0, 255)
        upper = np.clip(color + eps, 0, 255)
    
        mask |= cv2.inRange(frame, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask = mask)
    return mask, output

def select_blue(frame, colors, eps=1.1):
    mask = frame[..., 2] > (np.maximum(frame[..., 0], frame[..., 1]) * eps)
    mask = mask.astype(np.uint8) * 255
    output = cv2.bitwise_and(frame, frame, mask = mask)
    return mask, output

def get_bbox(mask):
    dilated_mask = cv2.morphologyEx(mask.copy(), op=cv2.MORPH_OPEN, kernel=np.ones((15,15)))
    x, y, w, h = cv2.boundingRect(dilated_mask)
    if h > 50 and w > 50:
        return (x, y, w, h)
    else:
        return None
    
def blue_detect(frames):
    result_boxes = []
    result_frames = []
    for frame in frames:
        _msk, _cup = select_blue(frame, [])
        bbox = get_bbox(_msk)
        result_boxes.append(bbox)
        result_frames.append(draw_bbox(frame, bbox))
    return result_frames, result_boxes

def draw_bbox(frame, box_xywh, show=False):
    if box_xywh is not None:
        x, y, w, h = box_xywh
        result = cv2.rectangle(frame.copy(), (x, y), (x+w, y+h), (0, 255, 0), 5)
    else:
        result = frame.copy()
    if show:
        from matplotlib import pyplot as plt
        plt.imshow(result)
    return result

def animate(frames):
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation
    import matplotlib
    from IPython.display import HTML, Video
    fig = plt.figure()
    ims = []
    for f in frames:
        im = plt.imshow(cv2.resize(f, dsize=(200, 110), interpolation=cv2.INTER_NEAREST), 
                        animated=True)
        ims.append([im])
    anim = animation.ArtistAnimation(fig, ims, interval=10, blit=True,
                                repeat_delay=1000)
    return HTML(anim.to_jshtml())

def load_yolo(yolo_dir):
    with open(os.path.join(yolo_dir, 'coco.names'), 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet(os.path.join(yolo_dir, 'yolov3.weights'), 
                          os.path.join(yolo_dir, 'yolov3.cfg'))
    return net, classes

def prepare_input(frames):
    hw_pairs = [(frame.shape[0], frame.shape[1]) for frame in frames]
    scale = 1/255 
    blob = cv2.dnn.blobFromImages(frames, scale, (416,416), (0,0,0), False, crop=False)
    return blob, hw_pairs

def predict(net, frames):
    def get_output_layers(net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    
    blob, hw_pairs = prepare_input(frames)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    return outs, hw_pairs

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    res = cv2.rectangle(img.copy(), (x,y), (x_plus_w,y_plus_h), color, 5)
    #print(label)
    return cv2.putText(res, label, (x+10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def find_bbox(net_outs, H, W, class_idx, conf_threshold = 0.02, nms_threshold = 0.4):
    class_ids = []
    confidences = []
    boxes = []

    for out in net_outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == class_idx:
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    #print(boxes)
    if len(boxes) == 0:
        return None
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    box_conf_pairs = [(boxes[i[0]], confidences[i[0]]) for i in indices]
    best_box, best_conf = max(box_conf_pairs, key=lambda p: p[1])
    #print(best_box, best_conf)
    return best_box

def int_bbox(bbox_xyhw, H, W):
    if bbox_xyhw is None:
        return None
    x, y, w, h = bbox_xyhw
    x = int(np.clip(x, 0, W))
    y = int(np.clip(y, 0, H))
    
    w = int(np.clip(w, 0, W - x))
    h = int(np.clip(h, 0, H - y))
    return [x, y, w, h]

def slice_outs(all_outs, i):
    if len(all_outs[0].shape) == 2:
        return all_outs
    return [out[i] for out in all_outs]

class YoloDetector:
    def __init__(self, net, class_idx, conf_threshold = 0.2, nms_threshold = 0.4):
        self.net = net
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_idx = class_idx
        
    def __call__(self, frames):
        all_outs, hw_pairs = predict(self.net, frames)
        result_boxes = []
        result_frames = []
        for i, (frame, (H, W)) in enumerate(zip(frames, hw_pairs)):
            outs = slice_outs(all_outs, i)
            best_box = find_bbox(outs, H, W, class_idx=self.class_idx, 
                                 conf_threshold=self.conf_threshold, nms_threshold=self.nms_threshold)
            best_box = int_bbox(best_box, H, W)
            result_boxes.append(best_box)
            result_frames.append(draw_bbox(frame, best_box))
        return result_frames, result_boxes

def process_video(input_file, detector_type, result_file):
    frames = get_all_frames(input_file)
    
    if detector_type == YOLO_DETECTOR:
        net, classes = load_yolo('yolo/')
        CUP_CLASS_IDX = classes.index('cup')
        detector = YoloDetector(net, class_idx=CUP_CLASS_IDX)
    elif detector_type == BLUE_DETECTOR:
        detector = blue_detect

    res_frames, res_bboxes = detector(frames)
    
    #utils.save_video(res_frames, result_file)
    save_results(input_file, res_frames, res_bboxes)
    
def ensure_video_precessed(input_fpath):
    out_dir = get_out_dir(input_fpath)
    boxes_file  = os.path.join(out_dir, 'boxes.json')
    if not os.path.exists(boxes_file):
        process_video(input_fpath, DEFAULT_DETECTOR, '')
    print("done")
        
def get_out_dir(input_fpath):
    return f'{input_fpath}_out'

def get_switches(input_fpath):
    out_dir = get_out_dir(input_fpath)
    with open(os.path.join(out_dir, 'boxes.json'), 'rt') as f:
        boxes_js = json.load(f)
    
    prev_state = None
    switch_fnames, switch_captions = [], []
    for fname in sorted(boxes_js.keys()):
        cur_state = len(boxes_js[fname]) > 0
        if prev_state is None:
            prev_state = cur_state
            continue
        if cur_state != prev_state:
            switch_fnames.append(os.path.join(out_dir, fname))
            switch_captions.append('появилась' if cur_state else 'пропала')
        prev_state = cur_state
    return switch_fnames, switch_captions