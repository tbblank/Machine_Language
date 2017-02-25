from random 						import random;
from pybrain.datasets 				import ClassificationDataSet;
from pybrain.tools.shortcuts		import buildNetwork;
from pybrain.structure.modules		import SoftmaxLayer;
from pybrain.supervised.trainers	import BackpropTrainer;
from pybrain.utilities				import percentError;
from math 							import sqrt;
from math 							import pow;
from pylab							import ion, ioff, figure, draw, contourf, clf, hold, plot


#Build the random dataset and add to the Classification Data Set structure
ds = ClassificationDataSet(2, 1);
for i in range(1000):
	x = random();
	y = random();
	result = 1; #do the coordinates lie on or in the circle
	#Calculate distance between two points, if greater than the acceptable radius mark with 0
	if(sqrt(pow((.5-x),2) + pow((.5-y),2))) > .4:
		result = 0;
	tup = x,y;
	ds.addSample(tup, result);  

#split dataset into training (75%) and test data (25%)
tstdata, trndata = ds.splitWithProportion(0.25);
#trndata._convertToOneOfMany();
#tstdata._convertToOneOfMany();

#build the network
fnn = buildNetwork(trndata.indim, 12, trndata.outdim, outclass=SoftmaxLayer);
#train the network
trainer = BackpropTrainer(fnn, dataset=trndata, momentum=.9, verbose=True, weightdecay=.01);
trainer.trainEpochs(10);

#figure(1);
#ioff();
#clf();
#hold(True);

print(percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class']));















