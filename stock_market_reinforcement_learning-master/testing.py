from market_model_builder import MarketModelBuilder
from keras.optimizers import SGD
from environment import MarketEnv
import numpy as np

model_filename = "model.h5"
model = MarketModelBuilder(model_filename).getModel()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer='rmsprop')

model.load_weights("model.h5")

env = MarketEnv(target_symbols=["1"], input_symbols=[],
                start_date="2015/04/27/",
                end_date="2018/04/27",
                sudden_death=-1.0)

data = env.get_data(symbol="1")
data_list = list(data.values())


prediction = []
for j in range(len(data_list)):
    if j < 59:
        continue

    state = []
    subject = []
    subject_vol = []
    for i in range(60):
        subject.append([data_list[j - i][2]])
        subject_vol.append([data_list[j - i][3]])
    state.append([[1, 0, 1]])
    state.append([[subject, subject_vol]])
    state = [np.array(i) for i in state]

    act = np.argmax(np.squeeze(model.predict(state),axis=0))
    prediction.append(act)
    print(act)

np.save("1",prediction)



# state = []
#
# state.append([[1, 0, 0]])
#
# subject = []
# subjectVolume = []
#
# for i in range(60):
#     try:
#         subject.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][2]])
#         subjectVolume.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][3]])
#     except Exception as e:
#         print((self.targetCode, self.currentTargetIndex, i, len(self.targetDates)))
#         self.done = True
# tmpState.append([[subject, subjectVolume]])
#
# tmpState = [array(i) for i in tmpState]
#
#
#
# prediction = model.predict(data)
# print(prediction)
