/*
 File: ContFramePool.C

 Author:
 Date  :

 */

/*--------------------------------------------------------------------------*/
/*
 POSSIBLE IMPLEMENTATION
 -----------------------

 The class SimpleFramePool in file "simple_frame_pool.H/C" describes an
 incomplete vanilla implementation of a frame pool that allocates
 *single* frames at a time. Because it does allocate one frame at a time,
 it does not guarantee that a sequence of frames is allocated contiguously.
 This can cause problems.

 The class ContFramePool has the ability to allocate either single frames,
 or sequences of contiguous frames. This affects how we manage the
 free frames. In SimpleFramePool it is sufficient to maintain the free
 frames.
 In ContFramePool we need to maintain free *sequences* of frames.

 This can be done in many ways, ranging from extensions to bitmaps to
 free-lists of frames etc.

 IMPLEMENTATION:

 One simple way to manage sequences of free frames is to add a minor
 extension to the bitmap idea of SimpleFramePool: Instead of maintaining
 whether a frame is FREE or ALLOCATED, which requires one bit per frame,
 we maintain whether the frame is FREE, or ALLOCATED, or HEAD-OF-SEQUENCE.
 The meaning of FREE is the same as in SimpleFramePool.
 If a frame is marked as HEAD-OF-SEQUENCE, this means that it is allocated
 and that it is the first such frame in a sequence of frames. Allocated
 frames that are not first in a sequence are marked as ALLOCATED.

 NOTE: If we use this scheme to allocate only single frames, then all
 frames are marked as either FREE or HEAD-OF-SEQUENCE.

 NOTE: In SimpleFramePool we needed only one bit to store the state of
 each frame. Now we need two bits. In a first implementation you can choose
 to use one char per frame. This will allow you to check for a given status
 without having to do bit manipulations. Once you get this to work,
 revisit the implementation and change it to using two bits. You will get
 an efficiency penalty if you use one char (i.e., 8 bits) per frame when
 two bits do the trick.

 DETAILED IMPLEMENTATION:

 How can we use the HEAD-OF-SEQUENCE state to implement a contiguous
 allocator? Let's look a the individual functions:

 Constructor: Initialize all frames to FREE, except for any frames that you
 need for the management of the frame pool, if any.

 get_frames(_n_frames): Traverse the "bitmap" of states and look for a
 sequence of at least _n_frames entries that are FREE. If you find one,
 mark the first one as HEAD-OF-SEQUENCE and the remaining _n_frames-1 as
 ALLOCATED.

 release_frames(_first_frame_no): Check whether the first frame is marked as
 HEAD-OF-SEQUENCE. If not, something went wrong. If it is, mark it as FREE.
 Traverse the subsequent frames until you reach one that is FREE or
 HEAD-OF-SEQUENCE. Until then, mark the frames that you traverse as FREE.

 mark_inaccessible(_base_frame_no, _n_frames): This is no different than
 get_frames, without having to search for the free sequence. You tell the
 allocator exactly which frame to mark as HEAD-OF-SEQUENCE and how many
 frames after that to mark as ALLOCATED.

 needed_info_frames(_n_frames): This depends on how many bits you need
 to store the state of each frame. If you use a char to represent the state
 of a frame, then you need one info frame for each FRAME_SIZE frames.

 A WORD ABOUT RELEASE_FRAMES():

 When we releae a frame, we only know its frame number. At the time
 of a frame's release, we don't know necessarily which pool it came
 from. Therefore, the function "release_frame" is static, i.e.,
 not associated with a particular frame pool.

 This problem is related to the lack of a so-called "placement delete" in
 C++. For a discussion of this see Stroustrup's FAQ:
 http://www.stroustrup.com/bs_faq2.html#placement-delete

 */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/* DEFINES */
/*--------------------------------------------------------------------------*/

#define KB *(0x1 << 10)

/*--------------------------------------------------------------------------*/
/* INCLUDES */
/*--------------------------------------------------------------------------*/

#include "cont_frame_pool.H"
#include "console.H"
#include "utils.H"
#include "assert.H"

/*--------------------------------------------------------------------------*/
/* DATA STRUCTURES */
/*--------------------------------------------------------------------------*/

/* -- (none) -- */

/*--------------------------------------------------------------------------*/
/* CONSTANTS */
/*--------------------------------------------------------------------------*/

/* -- (none) -- */

/*--------------------------------------------------------------------------*/
/* FORWARDS */
/*--------------------------------------------------------------------------*/

ContFramePool *ContFramePool::frame_pool_head;
ContFramePool *ContFramePool::frame_pool_list;

/*--------------------------------------------------------------------------*/
/* METHODS FOR CLASS   C o n t F r a m e P o o l */
/*--------------------------------------------------------------------------*/

ContFramePool::ContFramePool(unsigned long _base_frame_no,
                             unsigned long _n_frames,
                             unsigned long _info_frame_no)
{
    assert(_n_frames <= FRAME_SIZE * 8);

    base_frame_no = _base_frame_no;
    nframes = _n_frames;
    nFreeFrames = _n_frames;

    if (_info_frame_no == 0)
    {
        bitmap = (unsigned char *)(base_frame_no * FRAME_SIZE);
    }
    else
    {
        bitmap = (unsigned char *)(_info_frame_no * FRAME_SIZE);
    }

    for (unsigned int i = 0; i * 8 < nframes * 2; i++)
    {
        bitmap[i] = 0x00;
    }

    if (_info_frame_no == 0)
    {
        bitmap[0] = 0x40;
        nFreeFrames--;
    }

    if (ContFramePool::frame_pool_head == NULL)
    {
        ContFramePool::frame_pool_head = this;
        ContFramePool::frame_pool_list = this;
    }
    else
    {
        ContFramePool::frame_pool_list->frame_pool_next = this;
        ContFramePool::frame_pool_list = this;
    }

    frame_pool_next = NULL;

    Console::puts("Frame Pool initialized!\n");
}

unsigned long ContFramePool::get_frames(unsigned int _n_frames)
{
    unsigned int need_frames = _n_frames;
    unsigned int frame_no = base_frame_no;
    int search = 0;
    int found = 0;
    int i_idx = 0;
    int j_idx = 0;

    if (_n_frames > nFreeFrames)
    {
        Console::puts("Exist frames that not available");
    }

    for (unsigned int i = 0; i < nframes / 4; i++)
    {
        unsigned char value = bitmap[i];
        unsigned char mask = 0xC0;
        for (int j = 0; j < 4; j++)
        {
            if ((bitmap[i] & mask) == 0)
            {
                if (search == 1)
                {
                    need_frames--;
                }
                else
                {
                    search = 1;
                    frame_no += i * 4 + j;
                    i_idx = i;
                    j_idx = j;
                    need_frames--;
                }
            }
            else
            {
                if (search == 1)
                {
                    frame_no = base_frame_no;
                    need_frames = _n_frames;
                    i_idx = 0;
                    j_idx = 0;
                    search = 0;
                }
            }
            mask = mask >> 2;
            if (need_frames == 0)
            {
                found = 1;
                break;
            }
        }
        if (need_frames == 0)
        {
            found = 1;
            break;
        }
    }

    if (found == 0)
    {
        Console::puts("No free frame found!");
        return 0;
    }

    int set_frame = _n_frames;
    unsigned char head_mask = 0x40;
    unsigned char inv_mask = 0xC0;
    head_mask = head_mask >> (j_idx * 2);
    inv_mask = inv_mask >> (j_idx * 2);
    bitmap[i_idx] = (bitmap[i_idx] & ~inv_mask) | head_mask;

    j_idx++;
    set_frame--;

    unsigned char a_mask = 0xC0;
    a_mask = a_mask >> (j_idx * 2);
    while (set_frame > 0 && j_idx < 4)
    {
        bitmap[i_idx] = bitmap[i_idx] | a_mask;
        a_mask = a_mask >> 2;
        set_frame--;
        j_idx++;
    }

    for (int i = i_idx + 1; i < nframes / 4; i++)
    {
        a_mask = 0xC0;
        for (int j = 0; j < 4; j++)
        {
            if (set_frame == 0)
            {
                break;
            }
            bitmap[i] = bitmap[i] | a_mask;
            a_mask = a_mask >> 2;
            set_frame--;
        }
        if (set_frame == 0)
        {
            break;
        }
    }

    if (search == 1)
    {
        nFreeFrames -= _n_frames;
        return frame_no;
    }
    else
    {
        Console::puts("No free frame found!");
        return 0;
    }
}

void ContFramePool::mark_inaccessible(unsigned long _base_frame_no,
                                      unsigned long _n_frames)
{
    unsigned long start = _base_frame_no;
    unsigned int bitmap_index;
    unsigned char marker;
    unsigned char marker_reset;
    for (start; start < (_base_frame_no + _n_frames); start++)
    {
        bitmap_index = ((start - base_frame_no) / 4);
        marker_reset = 0xc0 << ((start % 4) * 2);
        marker = 0x80 >> ((start % 4) * 2);
        bitmap[bitmap_index] = bitmap[bitmap_index] & (~marker_reset);
        bitmap[bitmap_index] = bitmap[bitmap_index] | marker;
    }
}

void ContFramePool::release_frames(unsigned long _first_frame_no)
{
    ContFramePool *cur_pool = ContFramePool::frame_pool_head;
    while ((cur_pool->base_frame_no > _first_frame_no) || (_first_frame_no >= cur_pool->base_frame_no + cur_pool->nframes))
    {
        cur_pool = cur_pool->frame_pool_next;
    }

    unsigned int bitmap_index = (_first_frame_no - cur_pool->base_frame_no) / 4;
    unsigned char checker_head = 0x80 >> (((_first_frame_no - cur_pool->base_frame_no) % 4) * 2);
    unsigned char checker_reset = 0xc0 >> (((_first_frame_no - cur_pool->base_frame_no) % 4 * 2));
    unsigned int i;
    if (((cur_pool->bitmap[bitmap_index] ^ checker_head) & checker_reset) == checker_reset)
    {
        cur_pool->bitmap[bitmap_index] = cur_pool->bitmap[bitmap_index] & (~checker_reset);
    }

    for (i = _first_frame_no; i < cur_pool->base_frame_no + cur_pool->nframes; i++)
    {
        int index = (i - cur_pool->base_frame_no) / 4;
        checker_reset = checker_reset >> (i - cur_pool->base_frame_no) % 4;
        if ((cur_pool->bitmap[index] & checker_reset) == 0)
        {
            break;
        }
        if ((cur_pool->bitmap[index] & checker_reset) == 0)
        {
            break;
        }
        cur_pool->bitmap[index] = cur_pool->bitmap[index] & (~checker_reset);
    }
}

unsigned long ContFramePool::needed_info_frames(unsigned long _n_frames)
{
    return (_n_frames * 2) / (8 * 4 KB) + ((_n_frames * 2) % (8 * 4 KB) > 0 ? 1 : 0);
}
