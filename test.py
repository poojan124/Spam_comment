from nb_classifier import classify

while True:
    s = input()
    if classify(s):
        print("Its spam!")
    else:
        print("Its not a spam.You are safe.")