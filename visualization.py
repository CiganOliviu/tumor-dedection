from model import history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_los'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()