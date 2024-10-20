from sklearn.utils import class_weight
from holo_cells.src.training.CustomsLoses import jaccard_loss,specificity, pixel_accuracy,dice_coef_loss,iou,focal_loss
import tensorflow as tf
import keras

from keras.callbacks import Callback
class CustomCallback(Callback):
      def on_train_batch_end(self, batch, logs=None):
          # Do any custom logic here if needed
          # self._update_progbar(batch, logs)  # This line is omitted to disable progbar update
          pass
          
def train(model, ruta_model, train_gen,val_gen, lr=1e-4, ee_patience= 17, w_dec=1e-6, reduce_patience=4 ):
    model.compile(
        optimizer=keras.optimizers.Adam(lr,weight_decay=w_dec), loss=jaccard_loss,#dice_coef_loss,#"binary_crossentropy",#1e-4,, , adamW,weight_decay=0.01
        metrics=[
        jaccard_loss,
        tf.keras.metrics.Recall() ,
        specificity,
        tf.keras.metrics.Precision(),
        pixel_accuracy,
        dice_coef_loss,
        tf.keras.metrics.BinaryIoU( target_class_ids=(0, 1), threshold=0.5, name=None, dtype=None),
        #iou,
        #focal_loss

    ]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(ruta_model, save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=ee_patience),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=reduce_patience, verbose=1),
        CustomCallback()
    ]

    # Train the model, doing validation at the end of each epoch.
    epochs = 200
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=2,
        
        #class_weight = {0:1,1:10} # aprentemente da problemas con algunos modelos
    )
    return history
