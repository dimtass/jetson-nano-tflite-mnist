# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MnistProt

import flatbuffers

class Stats(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsStats(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Stats()
        x.Init(buf, n + offset)
        return x

    # Stats
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Stats
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # Stats
    def Freq(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Stats
    def Mode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def StatsStart(builder): builder.StartObject(3)
def StatsAddVersion(builder, version): builder.PrependUint8Slot(0, version, 0)
def StatsAddFreq(builder, freq): builder.PrependUint32Slot(1, freq, 0)
def StatsAddMode(builder, mode): builder.PrependInt8Slot(2, mode, 0)
def StatsEnd(builder): return builder.EndObject()