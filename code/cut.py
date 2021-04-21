import const
import time
import os

start_time = time.time()

def cut(file_name: str):
    lines_per_file = 1000
    filename = "docs/" + file_name + ".txt"
    smallfile = None
    with open(filename, encoding="utf8") as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = filename + '_{}.txt'.format(lineno + lines_per_file)
                smallfile = open(small_filename, "w", encoding="utf8")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

def cut_clean():
    for file_name in const.books:
        cut(file_name)
        os.remove("docs/" + file_name + ".txt")

    print("CUT AND CLEAN Process finished --- %.6s seconds ---" % (time.time() - start_time))

def listFiles():
    onlyfiles = [f for f in os.listdir("docs/") if os.path.isfile(os.path.join("docs/", f))]
    smallfile = open("filelist.txt", "w", encoding="utf8")
    smallfile.write(str(onlyfiles))

def main():
    print("Call desired function down below")