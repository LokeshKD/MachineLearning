

## all functions in here


model = Sequential()
layer1 = Dense(5, input_dim=4)
model.add(layer1)
layer2 = Dense(3, activation='relu')
model.add(layer2)


### another way to achieve the above
layer1 = Dense(5, input_dim=4)
layer2 = Dense(3, activation='relu')
model = Sequential([layer1, layer2])


### binary classification :- sigmoid
model = Sequential()
layer1 = Dense(5, activation='relu', input_dim=4)
model.add(layer1)
layer2 = Dense(1, activation='sigmoid')
model.add(layer2)

### Multiclass classification :- softmax
model = Sequential()
layer1 = Dense(5, input_dim=4)
model.add(layer1)
layer2 = Dense(3, activation='softmax')
model.add(layer2)


## Config for tranining.
model = Sequential()
layer1 = Dense(5, activation='relu', input_dim=4)
model.add(layer1)
layer2 = Dense(1, activation='sigmoid')
model.add(layer2)
model.compile('adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

### Execution

model = Sequential()
layer1 = Dense(200, activation='relu', input_dim=4)
model.add(layer1)
layer2 = Dense(200, activation='relu')
model.add(layer2)
layer3 = Dense(3, activation='softmax')
model.add(layer3)
model.compile('adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# predefined multiclass dataset
train_output = model.fit(data, labels,
                         batch_size=20, epochs=5)


## Evaluate
# predefined eval dataset
print(model.evaluate(eval_data, eval_labels))


### Predictions
# 3 new data observations
print('{}'.format(repr(model.predict(new_data))))



