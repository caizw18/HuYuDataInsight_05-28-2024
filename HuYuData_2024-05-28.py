cifar = modelCIFAR()

use_cuda = True

if use_cuda and torch.cuda.is_available():
    cifar.cuda()


# 'ASGD': optim.ASGD(cifar.parameters(), lr)}
# 'Adam': optim.Adam(cifar.parameters(), lr)
# 'SGD': optim.SGD(cifar.parameters(), lr, momentum=0.9)


optimizer_dict = {'SGD': optim.SGD(cifar.parameters(), lr=learning_rate, momentum=0.9)}
# 'Adam': optim.Adam(cifar.parameters(), lr=learning_rate, weight_decay=weight_decay),
# 'ASGD': optim.ASGD(cifar.parameters(), lr=learning_rate, weight_decay=weight_decay)}
# optimizer_dict = {'Adam': optim.Adam(cifar.parameters(), lr=learning_rate, weight_decay=weight_decay)}
# optimizer_dict = {'ASGD': optim.ASGD(cifar.parameters(), lr=learning_rate, weight_decay=weight_decay)}
criterion = nn.CrossEntropyLoss()

def train_CIFAR():
    for epoch in range(epoch_range):
        runningLoss = 0.0
        trAcc = 0.0
        totTrain = 0
        start_time = time.time()
        for i, data in enumerate(train_set, 0):
            inputTrain, labelTrain = data
            inputTrain = Variable(inputTrain)
            labelTrain = Variable(labelTrain)

            if use_cuda and torch.cuda.is_available():
                inputTrain = inputTrain.cuda()
                labelTrain = labelTrain.cuda()

            optimizer.zero_grad()
            outputTrain = cifar(inputTrain)
            loss = criterion(outputTrain, labelTrain)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            trainEpoch.append(loss.item())

            totTrain += 1
            _, pred = torch.max(outputTrain, dim=1)
            correct_train = pred.eq(labelTrain.data.view_as(pred))
            accuracy_train = torch.mean(correct_train.type(torch.FloatTensor))
            trAcc += accuracy_train.item()

        with torch.no_grad():
            corr, test_l = test_CIFAR_1(cifar, test_set)

        trainLoss.append(runningLoss / totTrain)
        testLoss.append(test_l / len(testdataset))
        trainAccuracy.append(trAcc / totTrain)
        testAccuracy.append(corr / len(testdataset))

        print("Epoch: {}/{} |".format(epoch + 1, epoch_range),
              "Train loss: %.3f |" % (runningLoss / totTrain),
              "Train Accuracy: %.3f |" % (100 * trAcc / totTrain),
              "Test loss: %.3f |" % (test_l / len(testdataset)),
              "Test Accuracy: %.3f |" % (100 * corr / len(testdataset)),
              "Time/Epoch: %.3f sec|" % (time.time() - start_time))

        runningLoss = 0.0
        trAcc = 0.0
    print('Finished Training')
    print_CIFAR(trainAccuracy, testAccuracy, trainLoss, testLoss, key, trainEpoch)



def test_CIFAR_1(cifar, test_set):
    correcTest = 0
    testLoss = 0.0
    for data in test_set:
        imageTest, labelTest = data

        if use_cuda and torch.cuda.is_available():
            imageTest = imageTest.cuda()
            labelTest = labelTest.cuda()

        outputTest = cifar(imageTest)
        testLoss += criterion(outputTest, labelTest).item()

        _, predicted_test = torch.max(outputTest.data, 1)
        correcTest += (predicted_test == labelTest).sum().item()

    return correcTest, testLoss