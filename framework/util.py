import random
import string


class Utils(object):

    @classmethod
    def generateRandomString(cls, length):
        
        randomString = "".join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=length))
        
        return randomString        
        