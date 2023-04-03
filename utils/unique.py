import pandas as pd


df = pd.read_csv("../out_dir/compose_2/compose_2_output_detector.csv")
print(df)

df.drop_duplicates(subset="avg_l2", keep=False, inplace=True)

print(df)
