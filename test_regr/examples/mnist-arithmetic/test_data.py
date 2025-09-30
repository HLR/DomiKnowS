import pytest
import torch
from data import get_readers, SumBalanceDataset
import config

class TestSumBalanceDataset:
    def test_dataset_length(self, num_train):
        """Test that dataset returns correct length"""
        trainloader, _, _, _ = get_readers(num_train)
        assert len(trainloader) == num_train

    def test_dataset_item_structure(self, num_train):
        """Test that dataset items have correct structure"""
        trainloader, _, _, _ = get_readers(num_train)
        item = trainloader[0]
        
        assert 'pixels' in item
        assert 'summation' in item
        assert 'digit' in item
        assert 'eval' in item
        
        assert item['pixels'].shape == (2, 784)  # 2 images, flattened
        assert item['summation'].shape == (1, 1)
        assert len(item['digit']) == 2

    def test_summation_correctness(self, num_train):
        """Test that summation equals sum of digits"""
        trainloader, _, _, _ = get_readers(num_train)
        
        for i in range(min(5, len(trainloader))):  # Test first 5 items
            item = trainloader[i]
            digit_sum = item['digit'][0] + item['digit'][1]
            summation = item['summation'].item()
            assert digit_sum == summation

    def test_valid_digit_range(self, num_train):
        """Test that digits are in valid range [0-9]"""
        trainloader, _, _, _ = get_readers(num_train)
        
        for i in range(min(5, len(trainloader))):
            item = trainloader[i]
            for digit in item['digit']:
                assert 0 <= digit <= 9

    def test_valid_summation_range(self, num_train):
        """Test that summations are in valid range [0-18]"""
        trainloader, _, _, _ = get_readers(num_train)
        
        for i in range(min(5, len(trainloader))):
            item = trainloader[i]
            summation = item['summation'].item()
            assert 0 <= summation <= 18