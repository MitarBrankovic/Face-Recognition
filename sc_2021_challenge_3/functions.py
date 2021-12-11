import cv2

def make_hog(img):
    nbins = 9  # broj binova
    cell_size = (7, 7)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog


def get_emotions(train_image_paths, train_image_labels):
    anger = []
    contempt = []
    disgust = []
    happiness = []
    neutral = []
    sadness = []
    surprise = []
    

    for i in range(len(train_image_paths)):
        if train_image_labels[i] == 'anger':
            anger.append(train_image_paths[i])
        elif train_image_labels[i] == 'contempt':
            contempt.append(train_image_paths[i])
        elif train_image_labels[i] == 'disgust':
            disgust.append(train_image_paths[i])
        elif train_image_labels[i] == 'happiness':
            happiness.append(train_image_paths[i])
        elif train_image_labels[i] == 'neutral':
            neutral.append(train_image_paths[i])
        elif train_image_labels[i] == 'sadness':
            sadness.append(train_image_paths[i])
        elif train_image_labels[i] == 'surprise':
            surprise.append(train_image_paths[i])

    print("anger: ", len(anger), " contempt: ", len(contempt), " disgust: ", len(disgust), " happiness: ",len(happiness),
        " neutral: ",len(neutral), " sadness: ",len(sadness)," surprise: ",len(surprise))
    
    return anger,contempt,disgust,happiness,neutral, sadness, surprise


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))









