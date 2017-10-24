from typing import List

import numpy as np

from constants import BARCODE_V7, NUM_BARCODE_V7_TARGETS

class Barcode:
    '''
    GESTALT target array with spacer sequences
    v7 barcode from GESTALT paper Table S4 is unedited barcode
    initial barcode state equal to v7 by default
    '''
    def __init__(self,
            barcode: List[str]=BARCODE_V7,
            unedited_barcode: List[str]=BARCODE_V7,
            cut_sites: List[int] = [6] * NUM_BARCODE_V7_TARGETS):
        """
        @param barcode: the current state of the barcode
        @param unedited_barcode: the original state of the barcode
        @param cut_sites: offset from 3' end of target for Cas9 cutting
        """
        # The original barcode
        self.unedited_barcode = unedited_barcode
        # an editable copy of the barcode (as a list for mutability)
        self.barcode = list(barcode)
        self.cut_sites = cut_sites
        # number of targets
        self.n_targets = (len(self.barcode) - 1)//2
        # a list of target indices that have a DSB and need repair
        self.needs_repair = set()

        assert(self.n_targets == len(self.cut_sites))

    def get_active_targets(self):
        """
        @return the index of the targets that can be cut, e.g. the targets that have no DSBs and are unmodified
        """
        # TODO: right now this code is pretty inefficient. we might want to cache which targets are active
        matches = [self.unedited_barcode[2 * i + 1] == self.barcode[2 * i + 1] for i in range(self.n_targets) if i not in self.needs_repair]
        return np.where(matches)[0]

    def cut(self, target_idx):
        """
        Marks this target as having a DSB
        """
        self.needs_repair.add(target_idx)

    def indel(self, target1: int, target2: int, left_del_len: int=0, right_del_len: int=0, insertion: str=''):
        '''
        a utility function for deletion/insertion

        @param target1: index of target with cut
        @param target2: index of target with cut
        if target1 != target2, create an inter-target deletion, otherwise focal deletion
        @param left_del_len: number of nucleotides to delete to the left
        @param right_del_len: number of nucleotides to delete to the right
        @param insertion: sequence placed between target deletions
        '''
        # TODO: make this code more efficient
        # indices into the self.barcode list (accounting for the spacers)
        index1 = 1 + 2*min(target1, target2)
        index2 = 1 + 2*max(target1, target2)
        #  Deletermine which can cut
        cut_site = self.cut_sites[target1]
        # sequence left of cut
        left = ','.join(self.barcode[:index1 + 1])[:-cut_site]
        # barcode sections between the two cut sites, if inter-target
        if target2 == target1:
            center = ''
        else:
            # TODO: check logic
            maketrans = str.maketrans
            center = '-' * cut_site + ',' + ','.join(self.barcode[(index1 + 1):index2]).translate(maketrans('ACGTacgt', '-'*8)) + ',' + '-' * (len(self.barcode[index2]) - cut_site)
        # sequence right of cut
        right = ','.join(self.barcode[index2:])[len(self.barcode[index2]) - cut_site:]
        # left delete
        deleted = 0
        for position, letter in reversed(list(enumerate(left))):
            if deleted == left_del_len:
                break
            if letter is not ',':
                left = left[:position] + '-' + left[(position + 1):]
                deleted += 1
        # right delete
        deleted = 0
        for position, letter in enumerate(right):
            if deleted == right_del_len:
                break
            if letter is not ',':
                right = right[:position] + '-' + right[(position + 1):]
                deleted += 1
        # put it back together
        self.barcode = (left + insertion + center + right).split(',')
        # Update needs_repair
        self.needs_repair.difference(range(target1, target2 + 1))

    def events(self):
        '''return the list of observable indel events in the barcdoe'''
        events = []
        insertion_total = 0
        # find the indels
        for indel in re.compile('[-acgt]+').finditer(str(self)):
            start = indel.start() - insertion_total
            # find the insertions(s) in this indel
            insertion = ''.join(insertion.group(0) for insertion in re.compile('[acgt]+').finditer(indel.group(0)))
            insertion_total =+ len(insertion)
            end = indel.end() - insertion_total
            events.append((start, end, insertion))
        return events

    def __repr__(self):
        return str(''.join(self.barcode))
