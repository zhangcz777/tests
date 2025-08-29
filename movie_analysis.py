import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 一、加载 MovieLens 100k 数据集

u_user = pd.read_csv(
    'ml-100k/u.user',
    sep='|',
    names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
    encoding='latin-1'
)
u_item = pd.read_csv(
    'ml-100k/u.item',
    sep='|',
    header=None,
    encoding='latin-1',
    usecols=list(range(24))
)
u_item.columns = [
    'movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
u_data = pd.read_csv(
    'ml-100k/u.data',
    sep='\t',
    names=['user_id', 'movie_id', 'rating', 'timestamp'],
    encoding='latin-1'
)

# 二、数据合并 & 分类处理

data = pd.merge(pd.merge(u_data, u_user, on='user_id'), u_item, on='movie_id')
genre_cols = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
data_genre = data.melt(
    id_vars=['user_id', 'gender', 'rating'],
    value_vars=genre_cols,
    var_name='genre',
    value_name='is_genre'
)
data_genre = data_genre[data_genre['is_genre'] == 1].drop(columns='is_genre')

# 三、排名统计 + 图表

genre_stats = data_genre.groupby('genre')['rating'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
print("\n各类型电影的平均评分与标准差 (Top 10):\n", genre_stats.head(10))

df_stats = genre_stats.reset_index()
x_pos = np.arange(len(df_stats))
means = df_stats['mean'].values
stds = df_stats['std'].values

plt.figure(figsize=(12, 6))
plt.bar(x_pos, means, yerr=stds, capsize=5, align='center', alpha=0.8)
plt.xticks(x_pos, df_stats['genre'], rotation=45, fontsize=10)
plt.ylabel('平均评分')
plt.xlabel('电影类型')
plt.title('不同电影类型的平均评分及标准差')
plt.tight_layout()
plt.savefig('排名统计图.png', dpi=300)
plt.show()

# 四、性别分析

genre_gender_mean = data_genre.groupby(['genre', 'gender'])['rating'].mean().unstack()
print("\n按类型和性别计算平均评分:\n", genre_gender_mean.head())

genre_gender_mean.plot(kind='bar', width=0.8)
plt.xticks(rotation=45)
plt.ylabel('平均评分')
plt.title('不同类型电影的男女用户平均评分比较')
plt.tight_layout()
plt.savefig('性别均分.png', dpi=300)
plt.show()

# 五、分数分布图

plt.figure(figsize=(8, 5))
sns.histplot(data=data, x='rating', hue='gender', bins=[1, 2, 3, 4, 5, 6], multiple='dodge', shrink=0.8)
plt.xticks([1, 2, 3, 4, 5])
plt.xlabel('评分')
plt.ylabel('人数')
plt.title('男女评分分布对比')
plt.tight_layout()
plt.savefig('分数分布.png', dpi=300)
plt.show()

# 六、年份趋势图

data['year'] = data['release_date'].str[-4:].astype(float)
yearly_avg = data.groupby('year')['rating'].mean().dropna()

plt.figure(figsize=(8, 4))
plt.plot(yearly_avg.index, yearly_avg.values, marker='o')
plt.xlabel('上映年份')
plt.ylabel('平均评分')
plt.title('各年份电影平均评分变化趋势')
plt.tight_layout()
plt.savefig('年份趋势.png', dpi=300)
plt.show()

# 七、年龄段与电影类型评分分析

# 添加年龄分组
data['age_group'] = pd.cut(
    data['age'],
    bins=[0, 18, 25, 35, 45, 50, 60, 100],
    labels=['<18', '18-24', '25-34', '35-44', '45-49', '50-59', '60+']
)

# 拆分类型
data_genre_age = data.melt(
    id_vars=['user_id', 'age_group', 'rating'],
    value_vars=genre_cols,
    var_name='genre',
    value_name='is_genre'
)
data_genre_age = data_genre_age[data_genre_age['is_genre'] == 1]

# 分组统计平均评分
age_genre_rating = data_genre_age.groupby(['age_group', 'genre'])['rating'].mean().unstack()

# 可视化热力图
plt.figure(figsize=(12, 6))
sns.heatmap(age_genre_rating, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('不同年龄段对电影类型的评分偏好')
plt.xlabel('电影类型')
plt.ylabel('年龄段')
plt.tight_layout()
plt.savefig('年龄类型.png', dpi=300)
plt.show()

# 八、不同类型电影随时间的评分趋势

# 提取年份
data = data[data['release_date'].notnull()].copy()
data['year'] = data['release_date'].str.extract(r'(\d{4})')
data = data[data['year'].notnull()].copy()
data['year'] = data['year'].astype(int)

# 拆类型后加入年份
data_genre_year = data.melt(
    id_vars=['rating', 'year'],
    value_vars=genre_cols,
    var_name='genre',
    value_name='is_genre'
)
data_genre_year = data_genre_year[data_genre_year['is_genre'] == 1]

# 按类型和年份分组
trend = data_genre_year.groupby(['year', 'genre'])['rating'].mean().reset_index()

# 选择几个常见类型绘制趋势图
selected_genres = ['Drama', 'Comedy', 'Romance', 'Action', 'Sci-Fi']
plt.figure(figsize=(10, 6))
for g in selected_genres:
    genre_data = trend[trend['genre'] == g]
    if not genre_data.empty:
        plt.plot(genre_data['year'], genre_data['rating'], label=g)

plt.xlabel('年份')
plt.ylabel('平均评分')
plt.title('不同类型电影随时间的评分变化趋势')
plt.legend()
plt.tight_layout()
plt.savefig('类型趋势图.png', dpi=300)
plt.show()

# 九、职业群体对类型偏好分析（热力图）

data_genre_occ = data.melt(
    id_vars=['user_id', 'occupation', 'rating'],
    value_vars=genre_cols,
    var_name='genre',
    value_name='is_genre'
)
data_genre_occ = data_genre_occ[data_genre_occ['is_genre'] == 1]

occ_genre_rating = data_genre_occ.groupby(['occupation', 'genre'], observed=True)['rating'].mean().unstack()

plt.figure(figsize=(14, 6))
if not occ_genre_rating.empty:
    sns.heatmap(occ_genre_rating, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('不同职业群体对电影类型的评分偏好')
    plt.xlabel('电影类型')
    plt.ylabel('职业')
    plt.tight_layout()
    plt.savefig('职业类型图.png', dpi=300)
    plt.show()
else:
    print("职业群体评分矩阵为空，可能是分组后无数据。")

# 十、玫瑰图（极坐标柱状图）
stats = data_genre.groupby('genre')['rating'].agg(['mean', 'std', 'count']).sort_values(by='mean', ascending=False)

stats_reset = stats.reset_index()
genres = stats_reset['genre'].tolist()
scores = stats_reset['mean'].values

angles = np.linspace(0, 2 * np.pi, len(genres), endpoint=False)

# 为了封闭曲线，补上第一个点
angles_full = np.concatenate((angles, [angles[0]]))
scores_full = np.concatenate((scores, [scores[0]]))

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles_full, scores_full, 'o-', linewidth=2)
ax.fill(angles_full, scores_full, alpha=0.25)

# 设置角度标签为 genres（不要补）
ax.set_thetagrids(angles * 180 / np.pi, genres, fontsize=10)
ax.set_title('不同类型电影的平均评分（玫瑰图）', fontsize=14)
plt.tight_layout()
plt.savefig('类型评分图.png', dpi=300)
plt.show()
