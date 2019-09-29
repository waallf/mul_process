import urllib.request
import time
import xlrd
import os
import wget
import multiprocessing


def save(i, cell_11):
    try:
        start = time.time()
        print("begin {} {}".format(i, cell_11))
        dest_dir = r'./douyin_download_data/{}.mp4'.format(i)  # 文件名，包含文件格式
        if os.path.exists(dest_dir):
            print("exist {} {} {}".format(i, cell_11, time.time()-start))
            return
        f = urllib.request.urlopen(r''+cell_11)
        data = f.read()
        with open(dest_dir, 'wb') as code:
            code.write(data)
        print("ok {} {} {}".format(i, cell_11, time.time()-start))
    except Exception:
        print("error {} {}".format(i, cell_11))
        with open("./error.txt", mode="a") as f:
            f.write("{} {}\n".format(i, cell_11))
        pass
    pass


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    workbook = xlrd.open_workbook('./douyin_url.xlsx')
    booksheet = workbook.sheet_by_index(0)  # 用索引取第一个sheet

    for i in range(booksheet.nrows):
        cell_11 = booksheet.cell_value(i, 0).strip()
        pool.apply_async(save, args=(i, cell_11))
        pass

    pool.close()
    pool.join()
