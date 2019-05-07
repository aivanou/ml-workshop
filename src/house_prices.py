from fastai.basics import *
from fastai.tabular import *
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split


dep_var = 'SalePrice'

cat_names = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
             'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
             'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
             'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
             'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
             'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',  'TotRmsAbvGrd', 'Functional',
             'Fireplaces', 'FireplaceQu', 'GarageType',
             'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageQual',
             'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
             'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

cont_names = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
              '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'EnclosedPorch', '3SsnPorch',
              'ScreenPorch', 'PoolArea','MiscVal', 'WoodDeckSF', 'OpenPorchSF', 'GarageArea']




path = Path('/home/ubuntu/data/house-prices')

train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')


# remove NA values

def remove_na(df):
    for cont_col in cont_names:
        df[cont_col] = df[cont_col].fillna(df[cont_col].mean())

    for cat_col in cat_names:
        df[cat_col] = df[cat_col].fillna("#nocat#")
    return df

train_df = remove_na(train_df)
test_df = remove_na(test_df)


# encode categorical data

def categorify_train(df):
    categories = {}
    for cat_name in cat_names:
        df.loc[:,cat_name] = df.loc[:,cat_name].astype('category').cat.as_ordered()
        categories[cat_name] =  df[cat_name].cat.categories
    return categories

def categorify_test(df, categories):
    for cat_name in cat_names:
        df.loc[:,cat_name] = pd.Categorical(df[cat_name], categories=categories[cat_name], ordered=True)

categories = categorify_train(train_df)
categorify_test(test_df, categories)

def encode_cats(df):
    for cat_name in cat_names:
        df[cat_name] = df[cat_name].cat.codes
    return df

def decode_cats(df, categories):
    for cat_name in cat_names:
        df[cat_name] = df[cat_name].apply(lambda x: categories[cat_name][x])
    return df

train_df = encode_cats(train_df)
test_df = encode_cats(test_df)


# normalize continuous data

def normalize_train(df, e = 1e-6):
    means, stds = {}, {}
    for cont_name in cont_names:
        means[cont_name], stds[cont_name] = df.loc[:, cont_name].mean(), df.loc[:, cont_name].std()
        df.loc[:, cont_name] = (df.loc[:, cont_name] - means[cont_name])/ (e+stds[cont_name])
    return means, stds

def normalize_target(df, e = 1e-6):
    mean, std = df.loc[:, dep_var].mean(), df.loc[:, dep_var].std()
    df.loc[:, dep_var] = np.log(df.loc[:, dep_var])


# def normalize_target(df, e = 1e-6):
#     mean, std = df.loc[:, dep_var].mean(), df.loc[:, dep_var].std()
#     df.loc[:, dep_var] = (df.loc[:, dep_var] - mean)/ (e+std)


def normalize_test(df, means, stds, e = 1e-6):
    for cont_name in cont_names:
        df.loc[:, cont_name] = (df.loc[:, cont_name] - means[cont_name])/ (e+stds[cont_name])

means, stds = normalize_train(train_df)
normalize_target(train_df)
normalize_test(test_df, means, stds)

# split by cont/cat

train_cont = train_df[cont_names].values
train_cat = train_df[cat_names].values
train_target = train_df[dep_var].values
test_cont = test_df[cont_names].values
test_cat = test_df[cat_names].values

print(train_cont.shape, train_cat.shape, test_cont.shape, test_cat.shape, train_target.shape)


def tens(np_array):
    return torch.from_numpy(np_array)

# get target variable and transform to pytorch tensors

train_x_cont, valid_x_cont, train_x_cat, valid_x_cat, train_y, valid_y = train_test_split(train_cont, train_cat, train_target, test_size=0.1, random_state=42)

x_train_cont = tens(train_x_cont.astype(np.float32))
x_train_cat = tens(train_x_cat.astype(np.int64))
x_valid_cont = tens(valid_x_cont.astype(np.float32))
x_valid_cat = tens(valid_x_cat.astype(np.int64))
y_train = tens(train_y.astype(np.float32))
y_valid = tens(valid_y.astype(np.float32))
x_test_cont = tens(test_cont.astype(np.float32))
x_test_cat = tens(test_cat.astype(np.int64))

train_dataset = TensorDataset(x_train_cont, x_train_cat, y_train)
valid_dataset = TensorDataset(x_valid_cont, x_valid_cat, y_valid)
test_dataset = TensorDataset(x_test_cont, x_test_cat)

data = DataBunch.create(train_ds = train_dataset, valid_ds = valid_dataset, test_ds = test_dataset)

# Create embedding sizes for categorical variables

def create_emb(cat_size):
    min_size, max_size = 8, 30
    return int(max(min(cat_size/2, max_size), min_size))

def create_embds():
    embds = []
    for cat_name in cat_names:
        cat_size = len(categories[cat_name])
        embds.append((cat_size, create_emb(cat_size)))
    return embds

embs_size = create_embds()


# Define NN Module

class LinearModel(nn.Module):

    def __init__(self, embs, layers, out_size, y_range=None):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(n_in, n_out) for (n_in, n_out) in embs])
        self.in_emb = sum(e.embedding_dim for e in self.embeds)
        self.in_cont = len(cont_names)
        self.layers_sizes = [self.in_emb + self.in_cont] + layers + [out_size]
        self.y_range = y_range
        layers = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_sizes[:-1], self.layers_sizes[1:])):
            layers.append(nn.Linear(n_in, n_out))
            if i != len(self.layers_sizes) - 2:
                activation = nn.ReLU(inplace=True)
                layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat):
        x_embed = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeds)]
        x_embed = torch.cat(x_embed, -1)
        x = torch.cat([x_embed, x_cont], -1)
        out = self.layers(x)
        #         import pdb;pdb.set_trace()
        if self.y_range is not None:
            out = (self.y_range[1] - self.y_range[0]) * torch.sigmoid(out) + self.y_range[0]
        return out


loss_func = nn.MSELoss()

max_log_y = np.log(np.max(train_df[dep_var])*1.2)
y_range = torch.tensor([0, max_log_y], device=defaults.device)
model = LinearModel(embs_size,[100,50], 1, y_range=y_range).cuda()

x_cont,x_cat,y = next(iter(data.train_dl))
y_hat = model(x_cont, x_cat)
loss_func(y_hat,y)


def update_model(x_cont, x_cat, y, optimizer, model):
    y_hat = model(x_cont, x_cat)
    loss = loss_func(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def get_valid_losses(model, data):
    losses = []
    for x_cont, x_cat, y in data.valid_dl:
        with torch.no_grad():
            y_hat = model(x_cont, x_cat)
            loss = loss_func(y_hat, y)
            losses.append(loss.item())
    return losses

def run_epoch(model,optimizer , data):
    losses = []
    for x_cont, x_cat, y in data.train_dl:
        losses.append(update_model(x_cont, x_cat, y, optimizer, model))
    return losses

max_log_y = np.log(np.max(train_df[dep_var])*1.2)
y_range = torch.tensor([0, max_log_y], device=defaults.device)

model = LinearModel(embs_size,[100,50,25], 1).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def run_epochs(num_epochs):
    for epoch in range(num_epochs):
        losses = run_epoch(model, optimizer, data)
        valid_loss = np.average(get_valid_losses(model, data))
        print('Average loss: ', np.average(losses), 'valid loss: ', valid_loss)

run_epochs(10)


submit_df = pd.DataFrame()
submit_df['Id'] = test_df['Id']

test_preds = np.array([])
for x_cont,x_cat in iter(data.test_dl):
    preds = model(x_cont, x_cat)
    preds = preds.detach().cpu().numpy()
#     import pdb;pdb.set_trace()
    preds = np.squeeze(preds)
    test_preds = np.concatenate((test_preds, preds))

# test_preds=learn.get_preds(DatasetType.Test)
# submit_df[dep_var]=np.exp(test_preds[0].data).numpy().T[0]
# submit_df.to_csv(path/"submission.csv",index=False)

