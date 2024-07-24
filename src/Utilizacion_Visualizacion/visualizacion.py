
import json
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, fbeta_score
import matplotlib.patches as patches

def Visualiza_carpeta_real(detector, mostres):
    for carp in mostres:
        carpeta_reals = carp
        boxes, pred_masc =detector.get_images_predict_and_print(carpeta_reals+"/",print_times=True)

        plt.imshow(pred_masc)
        plt.show()

        #boxes sobre el holograma projectat
        holograma = carpeta_reals+"_hologram/"
        if os.path.exists(holograma):

            for f in os.listdir(holograma):
                if f.endswith(".tiff"):
                    im=Image.open(holograma + f)
                    plt.imshow(np.array(im))
                    plt.show()
                if f.endswith(".png"):

                    im = mpimg.imread(holograma + f)
                    plt.imshow(im)
                    plt.axis('off')
                    plt.show()

def evalua_imatge_real(detector, mostra, analisis_path, uid):

    with open(analisis_path, 'r') as file:
        analysis_data = json.load(file)

    def find_annotations_for_image(image_uid, analysis_data):
        for key, value in analysis_data.items():
            if 'image' in value and value['image'] == image_uid:
                return value.get('annotations', [])
        return []
    #uid="fdfdd43082d84eaa85823181ea05a98b"
    annotations = find_annotations_for_image(uid, analysis_data)

    for carp in mostra:
        carpeta_reals = carp
        boxes, pred_masc =detector.get_images_predict_and_print(carpeta_reals+"/",print_times=True)
        
        plt.imshow(pred_masc)
        plt.show()

        #boxes sobre el holograma projectat
        holograma = carpeta_reals+"_hologram/"
        if os.path.exists(holograma):
            for f in os.listdir(holograma):
                if f.endswith(".png"):

                    mostra = mpimg.imread(holograma + f)
                    plt.imshow(mostra)
                    plt.axis('off')
                    plt.show()
                if f.endswith("backprop0zEscalada.png"):

                    im = Image.open(holograma + f)
                    im=im.resize(( 2443,1958))
            fig, ax =  plt.subplots(1, figsize=(10, 10))
            ax.imshow(np.array(im))
            original_boxes=len(boxes)
            for box in boxes:
                x, y, x2, y2 = box
                rect = patches.Rectangle((x, y), x2-x, y2-y, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            annoted_boxes=0

            for annotation in annotations:
                bbox = annotation.get('bounding_box', [])
                if bbox:
                    annoted_boxes+=1
                    x, y, width, height = bbox
                    rect = patches.Rectangle((x+25, y+25), width, height, linewidth=1, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
            plt.show()
            print("original boxes:" +str(original_boxes))
            print("annoted boxes:" +str(annoted_boxes))


        def boxes_collide(box1, box2):
            x1, y1, x2, y2 = box1
            x1g, y1g, x2g, y2g = box2

            return not (x2 < x1g or x2g < x1 or y2 < y1g or y2g < y1)

        def generate_confusion_matrix(boxes, annotations):
            TP = 0
            FP = 0
            FN = 0

            for annotation in annotations:
                bbox = annotation.get('bounding_box', [])
                if bbox:
                    x, y, width, height = bbox
                    x += 25
                    y += 25
                    annotation_box = [x, y, x + width, y + height]

                    matched = False
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        detected_box = [x1, y1, x2, y2]

                        if boxes_collide(annotation_box, detected_box):
                            TP += 1
                            matched = True
                            break

                    if not matched:
                        FN += 1

            FP = len(boxes) - TP

            return np.array([[TP, FP], [FN, 0]])

        def calculate_metrics(confusion_matrix):
            TP = confusion_matrix[0, 0]
            FP = confusion_matrix[0, 1]
            FN = confusion_matrix[1, 0]

            precision = precision_score([1] * TP + [0] * FP, [1] * TP + [1] * FP)
            recall = recall_score([1] * TP + [1] * FN, [1] * TP + [0] * FN)
            f2_score = fbeta_score([1] * TP + [1] * (FP + FN), [1] * TP + [0] * (FP + FN), beta=2)

            return precision, recall, f2_score

        confusion_matrix = generate_confusion_matrix(boxes, annotations)
        precision, recall, f2_score = calculate_metrics(confusion_matrix)
        print(f"\nPrecision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F2 Score: {f2_score:.2f}")

        # Visualizing the confusion matrix using Seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Actual Positive", "Actual Negative"], yticklabels=["Predicted Positive", "Predicted Negative"])
        plt.title("Confusion Matrix")
        plt.ylabel("Predicted")
        plt.xlabel("Actual")
        plt.show()
