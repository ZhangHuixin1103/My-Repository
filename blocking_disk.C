/*
    File       : blocking_disk.c

    Author     : Huixin Zhang
    Modified   : 11/19/2022

    Description:

*/

/*--------------------------------------------------------------------------*/
/* DEFINES */
/*--------------------------------------------------------------------------*/

/* -- (none) -- */

/*--------------------------------------------------------------------------*/
/* INCLUDES */
/*--------------------------------------------------------------------------*/

#include "assert.H"
#include "utils.H"
#include "console.H"
#include "blocking_disk.H"
#include "scheduler.H"

extern Scheduler *SYSTEM_SCHEDULER;

/*--------------------------------------------------------------------------*/
/* CONSTRUCTOR */
/*--------------------------------------------------------------------------*/

BlockingDisk::BlockingDisk(DISK_ID _disk_id, unsigned int _size)
    : SimpleDisk(_disk_id, _size)
{
    head = NULL;
    tail = NULL;
    disk_id = _disk_id;
    SYSTEM_SCHEDULER->disk_register(this);
}

/*--------------------------------------------------------------------------*/
/* SIMPLE_DISK FUNCTIONS */
/*--------------------------------------------------------------------------*/

void BlockingDisk::issue_operation(DISK_OPERATION _op, unsigned long _block_no)
{

    Machine::outportb(0x1F1, 0x00);
    Machine::outportb(0x1F2, 0x01);
    Machine::outportb(0x1F3, (unsigned char)_block_no);
    Machine::outportb(0x1F4, (unsigned char)(_block_no >> 8));
    Machine::outportb(0x1F5, (unsigned char)(_block_no >> 16));
    Machine::outportb(0x1F6, ((unsigned char)(_block_no >> 24) & 0x0F) | 0xE0 | (disk_id << 4));
    Machine::outportb(0x1F7, (_op == READ) ? 0x20 : 0x30);
}

void BlockingDisk::enqueue()
{
    Request *request = new Request;
    request->thread = Thread::CurrentThread();
    request->next = NULL;
    if (tail != NULL)
    {
        tail->next = request;
        tail = request;
    }
    else
    {
        head = request;
        tail = request;
    }
}

void BlockingDisk::dequeue()
{
    Request *request = head;
    head = head->next;
    if (head == NULL)
        tail = NULL;
    delete request;
}

void BlockingDisk::read(unsigned long _block_no, unsigned char *_buf)
{
    enqueue();

    disk_status = true;
    SYSTEM_SCHEDULER->yield();
    issue_operation(READ, _block_no);
    disk_status = false;
    while (!is_ready())
    {
        SYSTEM_SCHEDULER->yield();
    }

    int i;
    unsigned short tmpw;
    for (i = 0; i < 256; i++)
    {
        tmpw = Machine::inportw(0x1F0);
        _buf[i * 2] = (unsigned char)tmpw;
        _buf[i * 2 + 1] = (unsigned char)(tmpw >> 8);
    }

    dequeue();
}

void BlockingDisk::write(unsigned long _block_no, unsigned char *_buf)
{
    enqueue();

    SYSTEM_SCHEDULER->yield();
    issue_operation(WRITE, _block_no);
    SYSTEM_SCHEDULER->yield();

    int i;
    unsigned short tmpw;
    for (i = 0; i < 256; i++)
    {
        tmpw = _buf[2 * i] | (_buf[2 * i + 1] << 8);
        Machine::outportw(0x1F0, tmpw);
    }

    dequeue();
}

Thread *BlockingDisk::request_head()
{
    if (head != NULL)
        return head->thread;
    else
        return NULL;
}

bool BlockingDisk::is_ready()
{
    return ((Machine::inportb(0x1F7) & 0x08) != 0);
}
