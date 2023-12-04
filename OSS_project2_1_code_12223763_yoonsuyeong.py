import pandas as pd

def func1(data_frame):
    years = [2015, 2016, 2017, 2018]
    columns = ['H', 'avg', 'HR', 'OBP']

    for year in years:
        print("*****", year, "년*****\n")
        for column in columns:
            year_data_frame = data_frame[data_frame['year'] == year]
            sorted_data_frame = year_data_frame.sort_values(by=column, ascending=False).head(10)
            top_ten_player = sorted_data_frame.head(10)
            print(column, "기준에 따른 상위 10명:")
            print(top_ten_player['batter_name'], "\n")

def func2(data_frame):
    columns = ['batter_name', 'war', 'cp']

    print("cp별 war가 가장 높은 선수들의 이름, war, cp \n")
    cp_data = data_frame.groupby('cp')
    result = data_frame.loc[cp_data['war'].idxmax()]
    
    print(result[columns])


def func3(data_frame):
    target_data_frame = data_frame[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']]
    correlation = target_data_frame.corrwith(data_frame['salary'])

    max_correlation = correlation.idxmax()

    print("salary와 가장 높은 correlation을 가진 변수 : ", max_correlation)
    print("해당 변수와 salary의 correlation : ", correlation[max_correlation])
        

if __name__=='__main__':
    data_frame = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    func1(data_frame)
    print("****************************************\n")
    func2(data_frame)
    print("\n****************************************\n")
    func3(data_frame)