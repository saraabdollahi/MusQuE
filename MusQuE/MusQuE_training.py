import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertConfig, BertModel, BertTokenizer
import torch
import itertools
import torch.nn as nn
from transformers import logging
import warnings
from torch import cuda
import random
import json
from rank_bm25 import BM25Okapi
import sys
import gc
import pickle
import glob


torch.manual_seed(42)
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

queries=pd.read_pickle("MusQuE/data/sample_event_queries.tsv", sep="\t", header=None, names=["eventkg_id","query","qid","q_text"])
all_queries=list(queries["query"].unique())
test_queries=random_sample(all_queries, int(len(all_queries)*0.2))
### train queries
all_queries=list(set(all_queries)-set(test_queries))

### save test_queries for Evaluation part
test_queries_df=pd.DataFrame(test_queries)
test_queries_df.to_csv("MusQuE/data/sample_test_queries.tsv", sep="\t", header=False, index=False)

terms_df=pd.read_pickle("/MusQuE/data/query_terms.tsv", header=None, names=["query","terms"])
terms_df["terms"]=terms_df["terms"].str.lower()

data=pd.read_csv("/MusQuE/data/sample_collection.tsv",sep="\t", header=None, names=["doc_id","content"])
neg_pos=pd.read_csv("/MusQuE/data/sample_top1000.tsv", sep="\t", header=None, names=["qid","neg","pos"])  
neg_pos_mrg=pd.merge(left=queries, right=neg_pos, how="left", on="qid")
neg_pos_mrg=neg_pos_mrg[["query","qid","pos"]].drop_duplicates()
query_docs=pd.merge(left=queries, right=neg_pos, how="left", on="qid")

class MusQuE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', max_length=512)
        self.MSE_loss=nn.MSELoss(size_average=True,reduction="mean")
        self.BCElogit_loss=nn.BCEWithLogitsLoss()
        self.sigmoid=nn.Sigmoid()
        self.cos=nn.CosineSimilarity(dim=(1,2), eps=1e-6)
        self.max_epoch=10
        self.LR=0.001
        self.number_of_stages=4
        self.linear = nn.Linear(768, 128, bias=False)
        self.bertcat_linear = nn.Linear(768, 1, bias=False)
        self.model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)
        self.colbert_model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)
        self.loss_weight1=nn.Parameter(torch.tensor(0.5))
        self.loss_weight2=nn.Parameter(torch.tensor(0.5))
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.bertcat_linear.weight)
        if self.bertcat_linear.bias is not None:
            nn.init.zeros_(self.bertcat_linear.bias)

    def query_bert_embedding(self,text): 
        tokenized_text = self.tokenizer(text, return_tensors="pt",max_length=200, padding="max_length")["input_ids"][0]
        segment=[1.0]*200
        segment_ids=torch.tensor(segment, requires_grad=True)
        tokenized_input=tokenized_text.unsqueeze(0)
        segment_input=segment_ids.unsqueeze(0)
        tokenized_input=tokenized_input.to(device)
        segment_input=segment_input.to(device)
        outputs = self.colbert_model(tokenized_input, segment_input)
        last_hidden_states=outputs[0][:, 0, :]    
        return(last_hidden_states)
        
    def tokenized_content(self, query):
        query_tmp=neg_pos_mrg.loc[neg_pos_mrg["query"]==query,]
        pos_docs=list(query_tmp["pos"].unique())
        if (len(pos_docs)==0):
            print(all_queries[j])
        tmp_df=query_docs.loc[query_docs["query"]==query,]
        tmp_df["neg"]=tmp_df["neg"].astype(int)
        tmp_df["pos"]=tmp_df["pos"].astype(int)
        all_docs=list(set(list(tmp_df["neg"].unique())+list(tmp_df["pos"].unique())))
        neg_docs=list(set(all_docs)-set(pos_docs))
        tmp_df_pos=data.loc[data["doc_id"].isin(pos_docs),]
        tmp_df_neg=data.loc[data["doc_id"].isin(neg_docs),]
        tmp_df_pos=tmp_df_pos.reset_index(drop=True)
        tmp_df_neg=tmp_df_neg.reset_index(drop=True)
        if (tmp_df_neg.shape[0]>1000):
            tmp_df_neg=tmp_df_neg.sample(frac=1).reset_index(drop=True)
            tmp_df_neg=tmp_df_neg.loc[:1000,:]
        final_dict={}
        final_dict[query]={}
        final_dict[query]["pos"]={}
        final_dict[query]["neg"]={}
        embedding_final_dict={}
        embedding_final_dict[all_queries[j]]={}
        tokenizer_output=tokenizer(tmp_df_pos.iloc[0]["content"], return_tensors="pt", max_length=200, padding="max_length", truncation="only_first")
        content_tensor_pos=tokenizer_output["input_ids"]
        content_attention_tensor_pos=tokenizer_output["attention_mask"]
        doc_ids=[]
        tmp_df_neg=tmp_df_neg.reset_index(drop=True)
        tmp_df_pos=tmp_df_pos.reset_index(drop=True)
        #print("tmp_df_pos: ", tmp_df_pos)
        for i in range(1,tmp_df_pos.shape[0]):
            tokenizer_output=tokenizer(tmp_df_pos.iloc[i]["content"], return_tensors="pt", max_length=200, padding="max_length", truncation="only_first")
            content=tokenizer_output["input_ids"]
            content_attention=tokenizer_output["attention_mask"]
            content_tensor_pos=torch.cat((content_tensor_pos, content), dim=0)
            content_attention_tensor_pos=torch.cat((content_attention_tensor_pos, content_attention), dim=0)
        final_dict[query]["pos"]["tokenized_documents"]=content_tensor_pos.numpy().tolist()
        final_dict[query]["pos"]["tokenized_attention_documents"]=content_attention_tensor_pos.numpy().tolist()
        final_dict[query]["pos"]["doc_ids"]=list(tmp_df_pos["doc_id"])
        tokenizer_output=tokenizer(tmp_df_neg.iloc[0]["content"], return_tensors="pt", max_length=200, padding="max_length", truncation="only_first")
        doc_ids=[]
        neg_size=tmp_df_neg.shape[0] 
        content_tensor_neg=tokenizer_output["input_ids"]
        content_attention_tensor_neg=tokenizer_output["attention_mask"]
        for k in range(1,tmp_df_neg.shape[0]):
            tokenizer_output=tokenizer(tmp_df_neg.iloc[k]["content"], return_tensors="pt", max_length=200, padding="max_length", truncation="only_first")
            content=tokenizer_output["input_ids"]
            content_attention=tokenizer_output["attention_mask"]
            content_tensor_neg=torch.cat((content_tensor_neg, content), dim=0)
            content_attention_tensor_neg=torch.cat((content_attention_tensor_neg, content_attention), dim=0)
        final_dict[query]["neg"]["tokenized_documents"]=content_tensor_neg.numpy().tolist()
        final_dict[query]["neg"]["tokenized_attention_documents"]=content_attention_tensor_neg.numpy().tolist()
        final_dict[query]["neg"]["doc_ids"]=list(tmp_df_neg["doc_id"])
        return(final_dict)
        
    
    def bert_embedding(self, contents, query):
        pos_doc_ids=(contents[query]["pos"]["doc_ids"])
        neg_doc_ids=(contents[query]["neg"]["doc_ids"]) 
        embedding_final_dict={}
        embedding_final_dict[query]={}
        tmp_bert_scores_df=pd.DataFrame() 
        contents_pos=contents[query]["pos"]["tokenized_documents"]
        contents_neg=contents[query]["neg"]["tokenized_documents"]
        pos_size=len(contents[query]["pos"]["tokenized_documents"])
        neg_size=len(contents[query]["neg"]["tokenized_documents"])
        pos_ids=contents[query]["pos"]["doc_ids"]
        neg_ids=contents[query]["neg"]["doc_ids"]
        all_ids=pos_ids+neg_ids
        content_pos_tensor=torch.tensor(contents_pos)
        content_pos_tensor=content_pos_tensor[:,:200]
        content_neg_tensor=torch.tensor(contents_neg)
        content_neg_tensor=content_neg_tensor[:,:200]
        document_segments_ids=[1] * (200)
        segments_ids=document_segments_ids
        segments_tensors = torch.tensor([segments_ids])
        pos_catbert_results=pd.DataFrame(columns=["number","score"])
        neg_catbert_results=pd.DataFrame(columns=["number","score"])
        model_content=content_pos_tensor[0,:].unsqueeze(0)
        model_segment=segments_tensors[0,:].unsqueeze(0)
        model_content=model_content.to(device)
        model_segment=model_segment.to(device)
        with torch.no_grad():
            tmp_output=self.model(model_content, model_segment)
        pos_tokens=tmp_output.hidden_states[-1]
        pos_tokens=self.linear(pos_tokens)
        pos_tokens=pos_tokens.to("cpu")
        embedding_tensor=pos_tokens
        for i in range(1, pos_size):
            model_content=content_pos_tensor[i,:].unsqueeze(0)
            model_segment=segments_tensors[0,:].unsqueeze(0)
            model_content=model_content.to(device)
            model_segment=model_segment.to(device)
            with torch.no_grad():
                tmp_output=self.model(model_content, model_segment)
            pos_tokens=tmp_output.hidden_states[-1]
            pos_tokens=self.linear(pos_tokens)
            pos_tokens=pos_tokens.to("cpu")
            embedding_tensor=torch.cat((embedding_tensor, pos_tokens), dim=0)
        model_content=content_neg_tensor[0,:].unsqueeze(0)
        model_segment=segments_tensors[0,:].unsqueeze(0)
        model_content=model_content.to(device)
        model_segment=model_segment.to(device)
        with torch.no_grad():
            tmp_output=self.model(model_content, model_segment)
        neg_tokens=tmp_output.hidden_states[-1]
        neg_tokens=self.linear(neg_tokens)
        neg_tokens=neg_tokens.to("cpu")
        embedding_tensor=torch.cat((embedding_tensor, neg_tokens), dim=0)
        step=200
        for w in range(1, neg_size, step):
            if(w+step>(neg_size)):
                content_neg_tensor_tmp=content_neg_tensor[w:neg_size,:]
                content_neg_tensor_tmp=content_neg_tensor_tmp.to(device)
                for i in range(0, neg_size-w):
                    model=self.model.to(device)
                    model_content=content_neg_tensor[i,:].unsqueeze(0)
                    model_segment=segments_tensors[0,:].unsqueeze(0)
                    model_content=model_content.to(device)
                    model_segment=model_segment.to(device)
                    with torch.no_grad():
                        tmp_output=self.model(model_content, model_segment)
                    neg_tokens=tmp_output.hidden_states[-1]
                    neg_tokens=self.linear(neg_tokens)
                    neg_tokens=neg_tokens.to("cpu")
                    embedding_tensor=torch.cat((embedding_tensor, neg_tokens), dim=0)
            else:
                content_neg_tensor_tmp=content_neg_tensor[w:w+step,:]
                content_neg_tensor_tmp=content_neg_tensor_tmp.to(device)
                for i in range(0, 200):
                    model_content=content_neg_tensor_tmp[i-1,:].unsqueeze(0)
                    model_segment=segments_tensors[0,:].unsqueeze(0)
                    model_content=model_content.to(device)
                    model_segment=model_segment.to(device)
                    with torch.no_grad():
                        tmp_output=self.model(model_content, model_segment)
                    neg_tokens=tmp_output.hidden_states[-1]
                    neg_tokens=self.linear(neg_tokens)
                    neg_tokens=neg_tokens.to("cpu")
                    embedding_tensor=torch.cat((embedding_tensor, neg_tokens), dim=0)
        embedding_final_dict[query]["doc_embeddings"]=embedding_tensor.detach().numpy().tolist()
        embedding_final_dict[query]["doc_ids"]=list(pos_doc_ids)+list(neg_doc_ids)
        return(embedding_final_dict)


    def subset_dict(self, query, my_dict, exclude_set):
        len_=len(my_dict[query]["doc_embeddings"])
        begin_embedding_list=my_dict[query]["doc_embeddings"][:10]
        begin_doc_ids_list=my_dict[query]["doc_ids"][:10]
        subset_indices=random.sample(set(range(len_))-set(exclude_set), k=90)
        exclude_indices=exclude_set+subset_indices
        random_embedding_list=[my_dict[query]["doc_embeddings"][i] for i in subset_indices]
        random_doc_ids_list=[my_dict[query]["doc_ids"][i] for i in subset_indices]
        new_embedding_list=begin_embedding_list+random_embedding_list
        new_doc_ids_list=begin_doc_ids_list+random_doc_ids_list
        subset_dict={}
        subset_dict[query]={}
        subset_dict[query]["doc_embeddings"]=new_embedding_list
        subset_dict[query]["doc_ids"]=new_doc_ids_list
        return(exclude_set, subset_dict)

    def late_score(self, document, query):
        query=query.unsqueeze(dim=0)
        output=torch.cdist(query, document, p=2) 
        score=output.amax(dim=1)
        final_score=score.sum(dim=1)
        min_=final_score.min()
        max_=final_score.max()
        final_score=(final_score-min_)/(max_-min_)
        return(final_score)
    
    def colbert(self,query, new_query, doc):
        query_embedding=self.query_bert_embedding(new_query)
        query_=self.linear(query_embedding)
        doc=doc.to(device)
        score=self.late_score(doc, query_)
        return(score)

    def difference_score(self,num, score, ids, bert_scores):
        score=score.to("cpu")
        colbert_score=pd.DataFrame(score.detach().numpy())
        colbert_score.columns={"score"}
        colbert_score=colbert_score.reset_index(drop=True)
        doc_ids=pd.DataFrame(ids)
        doc_ids.columns={"doc_id"}
        doc_ids=doc_ids.reset_index(drop=True)
        colbert_df=pd.concat([colbert_score,doc_ids], axis=1)
        colbert_df["rank1"]=colbert_df["score"].rank(method="first", ascending=False)
        bert_scores["rank2"]=bert_scores["score"].rank(method="first", ascending=False)
        mrg=pd.merge(left=bert_scores, right=colbert_df, how="inner", on="doc_id")
        mrg["diff_score"]=mrg["rank1"]-mrg["rank2"]
        sum_score=mrg["diff_score"].abs().sum()
        return(sum_score)

    def combination(self, terms, num):
        n = 2
        pairs_to_keep = 2+num
        pairs = list(itertools.combinations(terms,num))
        filtered_pairs = []
        for pair in pairs:
            count1 = sum(1 for p in pairs if pair[0] in p or pair[1] in p)
            count2 = sum(1 for p in filtered_pairs if pair[0] in p or pair[1] in p)
            if count1 >= n and count2 < pairs_to_keep:
                filtered_pairs.append(pair)
        final=[list(f) for f in filtered_pairs]
        return(final)


    def multi_stage_colbert(self,optimizer, query, data, stage, all_terms, bert_results, previous_best_score, previous_best_query):
        final_results=pd.DataFrame(columns=["query","doc_id","score"])
        query_scores=pd.DataFrame(columns=["query","score","query_tmp"])
        combination_terms=self.combination(all_terms, stage+1)
        final_results_dict={}
        doc=torch.tensor(data[query]["doc_embeddings"])
        for j in range(len(combination_terms)):
            new_query=query+" "+' '.join(combination_terms[j])
            new_query_tmp=query+" ,"+', '.join(combination_terms[j])
            score=self.colbert(query, new_query, doc)
            final_results_dict[new_query]=score  
            diff_score=self.difference_score(j,score, data[query]["doc_ids"],bert_results)
            query_scores=query_scores.append({"query":new_query, "score":diff_score,"query_tmp":new_query_tmp}, ignore_index=True)
        query_scores=query_scores.drop_duplicates()
        if (query_scores.shape[0]==0):
            stage_flag=0 
            return(1, all_terms, 10000000000000, stage_flag, previous_best_query)
        else:
            query_scores=query_scores.sort_values(by="score")
            query_scores=query_scores.reset_index(drop=True)
            if (stage==1):
                tmp_query=query_scores.head(8)
            elif (stage==2):
                tmp_query=query_scores.head(7)
            elif (stage==3):
                tmp_query=query_scores.head(6)
            elif (stage==4):
                tmp_query=query_scores.head(5)
            tmp_query["query_tmp"]=tmp_query["query_tmp"].str.replace(query+" ,", "")
            tmp_query["lis"]=tmp_query["query_tmp"].str.split(", ")
            tmp_query=tmp_query.explode("lis")
            terms=list(tmp_query["lis"].unique())
            best_query=query_scores.iloc[0]["query"]
            current_best_score=query_scores.iloc[0]["score"]
            
            if (current_best_score<previous_best_score):
                stage_flag=1
                best_doc_scores=final_results_dict[best_query]
                bert_results=bert_results.loc[bert_results["doc_id"].isin(doc_ids),]
                bert_grounds=list(bert_results["score"])
                bert_grounds_tensor=torch.tensor(bert_grounds)
                bert_grounds_tensor=bert_grounds_tensor.to(device)
                bert_grounds_tensor.requires_grad=True
                stage_loss=self.MSE_loss(best_doc_scores, bert_grounds_tensor)
                return(stage_loss,terms, current_best_score, stage_flag, best_query)
            else:
                stage_flag=0
                return(1, all_terms, current_best_score, stage_flag, previous_best_query)

    def final_stage(self,stage_flag, content_dict, query, data, terms, best_query):
        if (stage_flag==0):
            new_query=best_query
        else:
            new_query=query+" "+' '.join(terms)    
        all_scores, tmp_bert_scores_df, all_pos_neg_results, all_pos_neg_labels=self.catbert(query,new_query,content_dict)
        labels=torch.ones((pos_scores.size()))
        labels.requires_grad=True
        labels=labels.to(device)
        loss=self.BCElogit_loss(all_pos_neg_results, all_pos_neg_labels) 
        return(loss, all_scores, tmp_bert_scores_df)


    def catbert(self,query,new_query, contents):
        tmp_bert_scores_df=pd.DataFrame() 
        contents_pos=contents[query]["pos"]["tokenized_documents"]
        contents_neg=contents[query]["neg"]["tokenized_documents"]
        pos_size=len(contents[query]["pos"]["tokenized_documents"])
        neg_size=len(contents[query]["neg"]["tokenized_documents"])
        pos_ids=contents[query]["pos"]["doc_ids"]
        neg_ids=contents[query]["neg"]["doc_ids"]
        pos_attention=contents[query]["pos"]["tokenized_attention_documents"]
        neg_attention=contents[query]["neg"]["tokenized_attention_documents"]
        all_ids=pos_ids+neg_ids
        
        query_output=self.tokenizer(new_query, return_tensors="pt")
        query_tokens=self.tokenizer(new_query, return_tensors="pt")["input_ids"][0]
        query_attention=query_output["attention_mask"]
        query_attention_pos=query_attention.repeat(pos_size, 1)
        query_attention_neg=query_attention.repeat(neg_size, 1)
        query_tensor_pos=query_tokens.repeat(pos_size, 1)
        query_tensor_neg=query_tokens.repeat(neg_size, 1)
        content_pos_tensor=torch.cat([query_tensor_pos, torch.tensor(contents_pos)],-1)
        content_pos_tensor=content_pos_tensor.to(device)
        content_pos_tensor=content_pos_tensor[:,:512]
        
        attention_pos_tensor=torch.cat([query_attention_pos, torch.tensor(pos_attention)],-1)
        attention_pos_tensor=attention_pos_tensor.to(device)
        attention_pos_tensor=attention_pos_tensor[:,:512]
        attention_neg_tensor=torch.cat([query_attention_neg, torch.tensor(neg_attention)],-1)
        attention_neg_tensor=attention_neg_tensor.to(device)
        attention_neg_tensor=attention_neg_tensor[:,:512]
        content_neg_tensor=torch.cat([query_tensor_neg, torch.tensor(contents_neg)],-1)
        content_neg_tensor=content_neg_tensor.to(device)
        content_neg_tensor=content_neg_tensor[:,:512]
        query_segments_ids = [0] * (query_tokens.size()[0]) 
        document_segments_ids=[1] * (200)
        segments_ids=query_segments_ids+document_segments_ids
        segments_tensors = torch.tensor([segments_ids])
        segments_tensors=segments_tensors.to(device)
        pos_catbert_results=pd.DataFrame(columns=["number","score"])
        neg_catbert_results=pd.DataFrame(columns=["number","score"])
        pos_tensor=torch.tensor([0.0], requires_grad=True)
        pos_tensor=pos_tensor.to(device)
        for i in range(pos_size):
            with torch.no_grad():
                tmp_output=self.model(content_pos_tensor[i,:].unsqueeze(0), attention_mask=attention_pos_tensor[i,:].unsqueeze(0), token_type_ids=segments_tensors[0,:].unsqueeze(0))
                pos_last_hidden_states = tmp_output.last_hidden_state
                pos_score=(torch.transpose(pos_last_hidden_states[0,0,:], 0, -1))
            score=self.bertcat_linear(pos_score)
            pos_tensor=torch.cat((pos_tensor, score), dim=0)
            pos_catbert_results=pos_catbert_results.append({"number":i, "score":score.item(), "content":self.tokenizer.decode(content_pos_tensor[i,:])}, ignore_index=True)
        neg_tensor=torch.tensor([0.0], requires_grad=True)
        neg_tensor=neg_tensor.to(device)
        for i in range(neg_size):
            with torch.no_grad():
                tmp_output=self.model(content_neg_tensor[i,:].unsqueeze(0),attention_mask=attention_neg_tensor[i,:].unsqueeze(0),token_type_ids=segments_tensors[0,:].unsqueeze(0))
                neg_last_hidden_states = tmp_output.last_hidden_state
                neg_score=(torch.transpose(neg_last_hidden_states[0,0,:], 0, -1))
            score=self.bertcat_linear(neg_score)
            neg_tensor=torch.cat((neg_tensor, score), dim=0)
            neg_catbert_results=neg_catbert_results.append({"number":i, "score":score.item(), "content":self.tokenizer.decode(content_neg_tensor[i,:])}, ignore_index=True)
        pos_scores__=torch.tensor(pos_tensor[1].repeat(neg_size,1), requires_grad=True)
        neg_scores__=neg_tensor[1:]
        neg_min=neg_scores__.min()
        pos_min=pos_scores__.min()
        if neg_min>pos_min:
            min_=pos_min
        else:
            min_=neg_min
        neg_max=neg_scores__.max()
        pos_max=pos_scores__.max()
        if neg_max>pos_max:
            max_=neg_max
        else:
            max_=pos_max
        neg_scores__=(neg_scores__-min_)/(max_-min_)
        pos_scores__=(pos_scores__-min_)/(max_-min_)
        pos_scores_tmp=pos_scores__[0]
        all_scores=torch.cat((pos_tensor[1:], neg_tensor[1:]), dim=0)
        scores_min=all_scores.min()
        scores_max=all_scores.max()
        all_scores__=(all_scores-scores_min)/(scores_max-scores_min) 
        pos_result=torch.tensor(pos_catbert_results["score"], requires_grad=True)
        pos_labels=torch.ones(pos_tensor[1:].size()[0], requires_grad=True)
        neg_result=torch.tensor(neg_catbert_results["score"], requires_grad=True)
        neg_labels=torch.zeros(neg_tensor[1:].size()[0], requires_grad=True)
        all_pos_neg_results=torch.cat((pos_result, neg_result),-1)
        all_pos_neg_labels=torch.cat((pos_labels, neg_labels), -1)
        all_contents=list(pos_catbert_results.append(neg_catbert_results)["content"])
        all_results=list(pos_catbert_results.append(neg_catbert_results)["score"])
        all_scores__=all_scores__.to("cpu")
        all_scores_list=list(all_scores__.detach().numpy())
        tmp_bert_scores_df["score"]=all_scores_list
        tmp_bert_scores_df["doc_id"]=all_ids
        tmp_bert_scores_df["query"]=query
        tmp_bert_scores_df["label"]=all_pos_neg_labels.detach().numpy()
        all_pos_neg_labels=all_pos_neg_labels.to(device)
        return(all_scores, tmp_bert_scores_df, all_scores, all_pos_neg_labels)

t=MusQuE()
device="cuda" if cuda.is_available() else "cpu"
t=t.to(device)



def train_iteration(all_queries, bert_results,terms_df, bert_model, linear, bertcat_linear, model, optimizer, bert_scores_df, epoch):
    t.train()
    random.shuffle(all_queries)
    total_loss=torch.tensor(0.0)
    device="cuda" if cuda.is_available() else "cpu" 
    total_loss=total_loss.to(device)
    loss_cnt=0
    for q in range(len(all_queries)):
        try:
                optimizer.zero_grad()
                loss=torch.tensor(0.0)
                loss=loss.to(device)
                terms_df1=terms_df.loc[terms_df["query"]==all_queries[q],]
                terms=list(set(terms_df1["terms"].unique()))
                
                for term1 in terms:
                    for term2 in terms:
                        if ((term1 in term2) and (term1!=term2)):
                            try:
                                terms.remove(term1)
                            except:
                                continue
                for terms1 in terms:
                    if terms1 in all_queries[q]:
                        terms.remove(terms1)
                if all_queries[q] in terms:
                    terms.remove(all_queries[q])
                
                if (len(terms)<2):
                    new_query=all_queries[q]+" "+' '.join(terms)
                    new_query="[CLS] "+new_query+" [SEP]"
                    content_dict=tokenized_content(self, query)
                    data_tmp_=t.bert_embedding(content_dict, all_queries[q])
                    bert_results_=bert_results.loc[bert_results["query"]==all_queries[q],]   
                    final_stage_loss, new_bert_results, tmp_bert_scores_df=t.final_stage(0, content_dict,all_queries[q], data_tmp, terms, new_query)
         
                else:
                    try:
                        
                        content_dict=tokenized_content(self, query)
                        data_tmp_=t.bert_embedding(content_dict, all_queries[q])
                        bert_results_=bert_results.loc[bert_results["query"]==all_queries[q],]
                    except:
                        continue
                    for stage in range(1,t.number_of_stages):
                        if (stage==1):
                            new_exclude_list, data_tmp=t.subset_dict(all_queries[q],data_tmp_, [0,1,2,3,4,5,6,7,8,9])
                            local_loss,terms,current_best_score, stage_flag, best_query=t.multi_stage_colbert(optimizer, all_queries[q], data_tmp, stage, terms, bert_results_, 10000000000000, all_queries[q])
                            if (stage_flag==1):
                                loss = (loss + local_loss)/2
                            else:
                                break
                        else:
                            new_exclude_list, data_tmp=t.subset_dict(all_queries[q], data_tmp_, new_exclude_list)
                            local_loss,terms, current_best_score, stage_flag, best_query=t.multi_stage_colbert(optimizer, all_queries[q], data_tmp, stage, terms, bert_results_, current_best_score, best_query)
                            if (stage_flag==1):
                                loss = (loss + local_loss)/2 
                            else:
                                break
                    device="cuda" if cuda.is_available() else "cpu"
                    if (stage_flag==1):
                        final_stage_loss, new_bert_results, tmp_bert_scores_df=t.final_stage(stage_flag, content_dict,all_queries[q], data_tmp, terms, best_query)
                    else:
                        final_stage_loss, new_bert_results, tmp_bert_scores_df=t.final_stage(stage_flag, content_dict,all_queries[q], data_tmp, terms, best_query)
                bert_scores_df=bert_scores_df.append(tmp_bert_scores_df)
                loss_cnt+=1
                loss = t.loss_weight1*final_stage_loss + t.loss_weight2*loss
                total_loss += loss
                loss.backward()
                optimizer.step()    
        except:
    return (total_loss, bert_scores_df, loss_cnt)
    
        


def main(all_queries, terms_df):
    optimizer=torch.optim.Adam(t.parameters(),lr=t.LR)
    
    bert_scores_df=pd.DataFrame()
    bert_results=pd.read_csv("/MosQuE/Data/sample_doc_ranker_zero_scores.tsv", sep="\t") ### DocRanker_0 scores
    loss, new_bert_results, loss_cnt=train_iteration(all_queries, bert_results, terms_df, t.bert_model, t.linear, t.bertcat_linear, t.model, optimizer, bert_scores_df, 0)
    print("loss of this epoc is: ", loss/loss_cnt)
    for epoch in range(1,t.max_epoch):
        loss, new_bert_results, loss_cnt=train_iteration(all_queries, new_bert_results,terms_df, t.bert_model, t.linear, t.bertcat_linear, t.model, optimizer, bert_scores_df, epoch)
        print("loss of this epoc is: ", loss/loss_cnt)
    torch.save(t.state_dict(),"/MosQuE/Data/trained_model")


if __name__=="__main__":
    main(all_queries, terms_df)









