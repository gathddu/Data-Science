from pathlib import Path
from typing import Dict, List
from enum import Enum


class DataSource(Enum):

    TRANSACTIONS = "transactions"
    CLIENTS = "clients"
    VEHICLES = "vehicles"


class ColumnMapping:
    
    TRANSACTIONS = {
        'id_transacao': 'transaction_id',
        'id_cliente': 'client_id',
        'id_veiculo': 'vehicle_id',
        'tipo_operacao': 'operation_type',
        'data_transacao': 'transaction_date',
        'valor_transacao': 'amount',
        'desconto': 'discount',
        'forma_pagamento': 'payment_method',
        'status_transacao': 'status',
        'vendedor_responsavel': 'responsible_seller',
        'filial': 'branch'
    }
    
    CLIENTS = {
        'id_cliente': 'client_id',
        'tipo_cliente': 'client_type',
        'nome_razao_social': 'name',
        'cpf_cnpj': 'cpf_cnpj',
        'email': 'email',
        'telefone': 'phone',
        'cidade': 'city',
        'estado': 'state',
        'cep': 'postal_code',
        'data_nascimento_fundacao': 'birth_date',
        'genero': 'gender',
        'profissao_segmento': 'profession_segment',
        'renda_mensal_faturamento': 'monthly_income',
        'data_cadastro': 'registration_date'
    }
    
    VEHICLES = {
        'id_veiculo': 'vehicle_id',
        'marca': 'brand',
        'modelo': 'model',
        'categoria': 'category',
        'ano_fabricacao': 'year_manufactured',
        'ano_modelo': 'year_model',
        'cor': 'color',
        'tipo_veiculo': 'vehicle_type',
        'km': 'kilometers',
        'preco_custo': 'cost_price',
        'preco_venda': 'sale_price',
        'status_estoque': 'stock_status',
        'placa': 'license_plate',
        'data_entrada': 'entry_date',
        'data_saida': 'exit_date'
    }


class DataTypeSchema:
    
    TRANSACTIONS = {
        'transaction_id': 'string',
        'client_id': 'string',
        'vehicle_id': 'string',
        'operation_type': 'string',
        'transaction_date': 'datetime64',
        'amount': 'float64',
        'discount': 'float64',
        'payment_method': 'string',
        'status': 'string',
        'responsible_seller': 'string',
        'branch': 'string'
    }
    
    CLIENTS = {
        'client_id': 'string',
        'client_type': 'string',
        'name': 'string',
        'cpf_cnpj': 'string',
        'email': 'string',
        'phone': 'string',
        'city': 'string',
        'state': 'string',
        'postal_code': 'string',
        'birth_date': 'datetime64',
        'gender': 'string',
        'profession_segment': 'string',
        'monthly_income': 'float64',
        'registration_date': 'datetime64'
    }
    
    VEHICLES = {
        'vehicle_id': 'string',
        'brand': 'string',
        'model': 'string',
        'category': 'string',
        'year_manufactured': 'int64',
        'year_model': 'int64',
        'color': 'string',
        'vehicle_type': 'string',
        'kilometers': 'float64',
        'cost_price': 'float64',
        'sale_price': 'float64',
        'stock_status': 'string',
        'license_plate': 'string',
        'entry_date': 'datetime64',
        'exit_date': 'datetime64'
    }


class StandardizationRules:
    
    STATUS_MAP = {
        'ativo': 'active',
        'active': 'active',
        'inativo': 'inactive',
        'inactive': 'inactive',
        'cancelado': 'cancelled',
        'cancelled': 'cancelled',
        'suspenso': 'suspended',
        'suspended': 'suspended',
        'bloqueado': 'blocked',
        'blocked': 'blocked',
        'pendente': 'pending',
        'pending': 'pending',
    }
    
    PAYMENT_METHOD_MAP = {
        'crédito': 'credit_card',
        'credit': 'credit_card',
        'credit_card': 'credit_card',
        'débito': 'debit_card',
        'debit': 'debit_card',
        'debit_card': 'debit_card',
        'boleto': 'bank_transfer',
        'transferência': 'bank_transfer',
        'transfer': 'bank_transfer',
        'bank_transfer': 'bank_transfer',
        'dinheiro': 'cash',
        'cash': 'cash',
        'pix': 'pix',
    }


class PathConfig:
    
    def __init__(self):
        self.BASE_PATH = Path(__file__).parent.parent
        self.RAW_DATA_PATH = self.BASE_PATH / 'data' / 'raw'
        self.PROCESSED_DATA_PATH = self.BASE_PATH / 'data' / 'processed'
        self.GOLD_DATA_PATH = self.BASE_PATH / 'data' / 'gold'
        self.OUTPUT_PATH = self.BASE_PATH / 'output'
        self.LOGS_PATH = self.BASE_PATH / 'logs'
        
        for path in [self.PROCESSED_DATA_PATH, self.GOLD_DATA_PATH, self.OUTPUT_PATH, self.LOGS_PATH]:
            path.mkdir(parents=True, exist_ok=True)


class ETLConfig:
    
    def __init__(self):

        self.paths = PathConfig()
        
        self.column_mappings = ColumnMapping()
        
        self.schemas = DataTypeSchema()
        
        self.rules = StandardizationRules()
        
        self.DUPLICATE_SUBSET = {
            'transactions': None,
            'clients': ['client_id'],
            'vehicles': ['vehicle_id']
        }
        
        self.IQR_MULTIPLIER = 1.5
        
        self.DATE_FORMATS = [
            '%d/%m/%Y',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%d-%m-%Y',
            '%m/%d/%Y',
        ]
        
        self.LOG_LEVEL = 'INFO'
        self.LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

config = ETLConfig()
