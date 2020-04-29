import csv
import os
from xlsxwriter.workbook import Workbook

'''
将单个tsv文件转为xlsx文件
'''
def tsv_to_xlsx(tsv_path_name):
    # Add some command-line logic to read the file names.
    tsv_file = tsv_path_name
    xlsx_file = os.path.splitext(tsv_path_name)[0] + '.xlsx'

    # Create an XlsxWriter workbook object and add a worksheet.
    workbook = Workbook(xlsx_file)
    worksheet = workbook.add_worksheet()

    # Create a TSV file reader.
    tsv_reader = csv.reader(open(tsv_file, 'rt'), delimiter='\t')

    # Read the row data from the TSV file and write it to the XLSX file.
    for row, data in enumerate(tsv_reader):
        worksheet.write_row(row, 0, data)

    # Close the XLSX file.
    workbook.close()

def tsv_2_xlsx_dir(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            #判断后缀为tsv的
            if os.path.splitext(file)[1] == '.tsv':
                tsv_to_xlsx(root + "\\" + file)

if __name__=="__main__":
    tsv_2_xlsx_dir('C:\\Users\\shand\\Desktop\\test')
    

    