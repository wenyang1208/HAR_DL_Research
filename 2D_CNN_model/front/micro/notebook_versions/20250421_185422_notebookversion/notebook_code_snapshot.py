import random

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0
y = to_categorical(y, num_classes=5)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# early stopping for preventing overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

cnn = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

cnn.compile(Adam(learning_rate = 0.0001),  
            loss='categorical_crossentropy',
            metrics=['accuracy'])

history = cnn.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

val_loss, val_acc = cnn.evaluate(X_val, y_val) 
print(f"Final Validation Accuracy: {val_acc}")

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
notebook_saver.save_plot(name = 'plot')
plt.show()

# Plot confusion matrix:
# Get predictions from the model on the validation set
y_pred_probs = cnn.predict(X_val)

y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class indices

# Convert one-hot encoded true labels to class indices
y_true = np.argmax(y_val, axis=1)

# Reference: https://github.com/parisafm/CSI-HAR-Dataset/blob/main/CNN.py
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    notebook_saver.save_plot(name = 'cm')

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

np.set_printoptions(precision=2)

# plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=CATEGORIES, title='Normalized confusion matrix')
plt.show()

cnn.summary()

report = classification_report(y_true, y_pred, target_names=CATEGORIES)
print("Classification Report:\n")
print(report)

# # 1. Create the Feature Extractor Model from the trained CNN
# #    We take layers up to Flatten (index -3)
# if len(cnn.layers) < 3:
#     print("Error: CNN model is too shallow for feature extraction at Flatten layer.")
# else:
#     feature_extractor = Model(inputs=cnn.input,
#                               outputs=cnn.layers[-3].output, # Output of the Flatten layer
#                               name="CNN_Feature_Extractor")
#     feature_extractor.summary() # Show the structure of the extractor

#     # 2. Extract Features
#     print("\n--- Extracting features using the trained CNN ---")
#     print("Extracting features from training data...")
#     X_train_features = feature_extractor.predict(X_train)
#     print("Extracting features from validation data...")
#     X_val_features = feature_extractor.predict(X_val)

#     print(f"Shape of extracted features (train): {X_train_features.shape}") # e.g., (num_samples, num_flattened_features)
#     print(f"Shape of extracted features (validation): {X_val_features.shape}")

#     # 3. Apply PCA
#     n_components = 64 # Hyperparameter: Choose number of components (e.g., 50, 64, 128, or based on explained variance)
#     if X_train_features.shape[1] < n_components:
#         print(f"Warning: n_components ({n_components}) is >= number of features ({X_train_features.shape[1]}). Setting n_components to {X_train_features.shape[1] - 1}.")
#         n_components = X_train_features.shape[1] - 1 if X_train_features.shape[1] > 1 else 1


#     if n_components > 0: # Proceed only if PCA makes sense
#         pca = PCA(n_components=n_components, random_state=42)

#         print(f"\n--- Applying PCA to reduce dimensionality to {n_components} components ---")
#         print("Fitting PCA on extracted training features...")
#         pca.fit(X_train_features) # Fit ONLY on training data

#         print("Transforming features using PCA...")
#         X_train_pca = pca.transform(X_train_features)
#         X_val_pca = pca.transform(X_val_features)

#         print(f"Shape after PCA (train): {X_train_pca.shape}")
#         print(f"Shape after PCA (validation): {X_val_pca.shape}")
#         print(f"Explained variance ratio by {n_components} components: {np.sum(pca.explained_variance_ratio_):.4f}")


#         # 4. Train a Simpler Classifier (Simple MLP) on PCA features
#         print("\n--- Defining and Training a Simple MLP on PCA features ---")
#         mlp_on_pca = tf.keras.models.Sequential([
#             tf.keras.layers.Input(shape=(n_components,)), # Input shape is now n_components
#             tf.keras.layers.Dense(32, activation='relu'),   # Smaller Dense layer
#             tf.keras.layers.Dropout(0.3),                   # Optional Dropout
#             tf.keras.layers.Dense(len(CATEGORIES), activation='softmax') # Output layer
#         ], name="MLP_on_PCA_Features")

#         mlp_on_pca.compile(Adam(learning_rate = 0.001), # Can use slightly higher LR maybe
#                            loss='categorical_crossentropy',
#                            metrics=['accuracy'])

#         mlp_on_pca.summary()

#         # Use early stopping for the MLP as well
#         mlp_early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True) # Shorter patience maybe

#         history_mlp = mlp_on_pca.fit(X_train_pca, y_train, # Use PCA features and original one-hot labels
#                                      epochs=40, # Train for fewer epochs maybe
#                                      batch_size=64,
#                                      validation_data=(X_val_pca, y_val),
#                                      callbacks=[mlp_early_stopping])


#         # 5. Evaluate the MLP on PCA features
#         print("\n--- Evaluating the Simple MLP on PCA features ---")
#         val_loss_mlp, val_acc_mlp = mlp_on_pca.evaluate(X_val_pca, y_val)
#         print(f"MLP on PCA Features - Final Validation Loss: {val_loss_mlp}")
#         print(f"MLP on PCA Features - Final Validation Accuracy: {val_acc_mlp}")


#         # --- Optional: Plotting for MLP on PCA ---
#         print("\n--- Plotting MLP on PCA History ---")
#         plt.figure(figsize=(12, 5))
#         plt.subplot(1, 2, 1)
#         plt.ylabel('Loss', fontsize=14)
#         plt.plot(history_mlp.history['loss'], label='Training Loss')
#         plt.plot(history_mlp.history['val_loss'], label='Validation Loss')
#         plt.legend(loc='upper right')
#         plt.title('MLP on PCA Features Loss')

#         plt.subplot(1, 2, 2)
#         plt.ylabel('Accuracy', fontsize=14)
#         plt.plot(history_mlp.history['accuracy'], label='Training Accuracy')
#         plt.plot(history_mlp.history['val_accuracy'], label='Validation Accuracy')
#         plt.legend(loc='lower right')
#         plt.title('MLP on PCA Features Accuracy')
#         plt.tight_layout()
#         notebook_saver.save_plot(name='mlp_on_pca_plot')
#         plt.show()


#         # --- Optional: Confusion Matrix for MLP on PCA ---
#         print("\n--- Generating Confusion Matrix for MLP on PCA ---")
#         y_pred_probs_mlp = mlp_on_pca.predict(X_val_pca)
#         y_pred_mlp = np.argmax(y_pred_probs_mlp, axis=1)
#         # y_true is the same as before (from original y_val)

#         # Compute confusion matrix
#         cm_mlp = confusion_matrix(y_true, y_pred_mlp)

#         # Plot normalized confusion matrix
#         plot_confusion_matrix(cm_mlp, classes=CATEGORIES, title='MLP on PCA Features - Normalized Confusion Matrix')
#         plt.show()

#         # Classification Report for MLP on PCA
#         print("\n--- Classification Report for MLP on PCA Features ---")
#         report_mlp = classification_report(y_true, y_pred_mlp, target_names=CATEGORIES, zero_division=0)
#         print(report_mlp)

#     else: # Case where n_components <= 0
#          print("\nSkipping PCA and MLP training because n_components is not positive.")

# Final Comparison
# print("\n\n--- Final Accuracy Comparison ---")
# print(f"Original CNN Final Validation Accuracy: {val_acc:.4f}")
# if 'val_acc_mlp' in locals(): # Check if MLP was trained
#     print(f"MLP on PCA Features Validation Accuracy: {val_acc_mlp:.4f}")
# else:
#     print("MLP on PCA Features was not run.")

# print("\n--- Script Finished ---")

notebook_saver.save_notebook_code(run_start_index)
notebook_saver.save_model_summary(cnn)
notebook_saver.save_training_output(history, val_loss, val_acc)

