import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer 
import warnings

# Essa versão do pandas levanta um alerta ao utilizar pd.append()
warnings.filterwarnings("ignore")

data = pd.read_csv('FashionDataset.csv')
data

# Tirando uma coluna, pois não será necessário
data.drop('Unnamed: 0', axis=1,  inplace=True)
data.info()

#Podemos notar que há falta de valores, e que "Sizes" tem sua maior parte em valores inexistentes
#E que possui 1172 tamanhos diferentes para os produtos

# Analisando melhor o dataset
data.describe(include=object)

# Visualizar de forma cuidadosa pois será uma das feature chave para o sistema de recomendação
data['Sizes']

# É encontrado mais uma inconcistencia nos dados sendo 'Size:Error Size'
# Iremos tirar então do data frame 'Nan' e 'Size:Error Size'
problem_data = data[(data['Sizes'] == 'Nan') | (data['Sizes'] == 'Size:Error Size')].index
data = data.drop(problem_data, axis=0)

# Com o tamanho do data_set e seu numero de feature pode ser que haja valores duplicados

# Vereficar e excluir os valores duplicados
sum(data.duplicated())
data.drop_duplicates(keep='first', inplace=True)

# Analisando a variavel 'Deatils', podemos ver que há produtos que são separados apenas pela cor, não levaremos a cor em
# conta ao recomendar um produto pois se já houve a compra de um produto, irei considerar que não haverá a vontade de comprar outro com apenas outra cor

# Podemos perceber um variavel que desejamos tirar, 'cor'
pd.options.display.max_colwidth = 100
data['Deatils'].head(50)

# Um mesmo loop será feito em duas vezes pelo fato de que ao tirar a cor o numero de dados poderá reduzir, assim o próximo loop terá menos valores

# Restaurando o index pois houveram valores excluidos anteriormente
data.set_axis(range(len(data)), axis=0, inplace=True)

## Os loops levam um tempo, tem um arquivo "FashionDataset_clean_data"
## com os dados já alterados pelos loops

# o loop irá tirar a cor da variavel 'Deatils'
for i in range(len(data)):

    # Color é a variavel que irá conter os 'Deatils' do produto sem as cores
    Color = data['Deatils'][i].split()
    for x in range(2):
        Color.pop(-1)

    if Color[-1] == '-':
        Color.pop(-1)

    data['Deatils'][i] = ' '.join(Color)

#Tirando a variavel cor, haverá mais dados duplicados

# Vereficar e excluir os valores duplicados
print(sum(data.duplicated()))
data.drop_duplicates(keep='first', inplace=True)

# Restaurando o index pois houveram valores excluidos anteriormente
data.set_axis(range(len(data)), axis=0, inplace=True)

# Novo data frame que será armazenado os dados "limpos"
new_data = pd.DataFrame(columns=data.columns)

# Será feita a separação das categorias de tamanho dos produtos em "Sizes"
for i in range(len(data)):

    Sizes = data['Sizes'][i].replace('Size:',' ')
    Sizes = Sizes.replace(',','  ')
    Sizes = Sizes.split()

    # A parte do loop que irá separar as meidas em novas fileiras   
    for j in Sizes:

        data.iloc[i]['Sizes'] = j
        new_data = new_data.append(data.iloc[i])

# Fazendo a separação das medidas pode ter ocorrido em mais dados duplicados

# Vereficar e excluir os valores duplicados
print(sum(new_data.duplicated()))
new_data.drop_duplicates(keep='first', inplace=True)

# Restaurando o index pois houveram valores excluidos anteriormente
new_data.set_axis(range(len(new_data)), axis=0, inplace=True)

# Analisando novamente os dados obtidos
new_data.describe(include=object)

# Para uma deteriminada 'Deatils', há 204 valores em iguais em duas colunas diferentes
new_data[(new_data['Deatils'] == 'printed round neck rayon womens ethnic set')][['BrandName', 'Category']].value_counts()

# Podemos perceber que  nas 3 variaveis que destinguem um produto, elas são iguais, só nas variaveis de valor que elas diferem
new_data[(new_data['Deatils'] == 'printed round neck rayon womens ethnic set') & (new_data['Sizes'] == 'Small')]

# Portanto iremos excluir as variaver relacionada ao valor já que não à utilizaremos 
new_data.drop(['MRP',	'SellPrice',	'Discount',], axis=1, inplace=True)
new_data.drop_duplicates(keep='first', inplace=True)

# Arrumando o index já que foram excluido algumas fileiras
new_data.set_axis(range(len(new_data)), axis=0, inplace=True)

# Criar uma nova coluna que ira conter os parametros usados para recomendar produtos simalares 
new_data['Similar'] = new_data['Deatils'] + ' ' + new_data['Category']

# Nova coluna que ira diferenciar os produtos 
new_data.insert(0,'Product_ID',range(len(new_data)))

# Função Product_Interaction irá simular uma (compra, adicionado item no carrinho, pesquisa) e irá recomendar novos produtos com base nessa  interação
def Product_Interaction(Product_ID=None):

    # Caso não seja selecionado algum produto, a fim de teste haverá uma escolha aleatória
    if Product_ID == None:
        Product_ID = np.random.choice(range(len(new_data['Product_ID'])))

    # Só será recomendado produtos que são das mesmas medidas que o produto interegido 
    Produtos = new_data[(new_data['Sizes'] == new_data.iloc[Product_ID]['Sizes'])]


    # Ferramentas que farão a distinção de quão parecido são os produtos 
    cv = CountVectorizer().fit_transform(Produtos['Similar'])
    cs = cosine_similarity(cv)
    rank = pd.DataFrame(cs)

    # É adicionado novamente a variavel que identifica cada produto 
    rank.insert(0, 'Product_ID', Produtos['Product_ID'].values)

    # Irá selecionar os 5  produtos que mais se parece com os interagidos
    recomendações = rank[rank['Product_ID'] == Product_ID].drop('Product_ID', axis=1).transpose()
    recomendações = recomendações.sort_values(by=recomendações.iloc[:,0].name)[-6:-1]

    # Mostrando os Resultados
    print('Baseado em seu interesse pelo produto:', '\n')
    print(f'{new_data.iloc[Product_ID]["Deatils"]}.', 'Tamanho:', new_data.iloc[Product_ID]['Sizes'],'\n')
    print('Te recomendamos:','\n')
    return Produtos.iloc[recomendações.index][['BrandName',	'Deatils', 'Sizes',	'Category']]

Product_Interaction(Product_ID=600)
Product_Interaction()
Product_Interaction()