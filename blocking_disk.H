/*
    File       : blocking_disk.H

    Author     : Huixin Zhang

    Date       : 11/19/2022
    Description:

*/

#ifndef _BLOCKING_DISK_H_
#define _BLOCKING_DISK_H_

/*--------------------------------------------------------------------------*/
/* DEFINES */
/*--------------------------------------------------------------------------*/

/* -- (none) -- */

/*--------------------------------------------------------------------------*/
/* INCLUDES */
/*--------------------------------------------------------------------------*/

#include "simple_disk.H"
#include "thread.H"

/*--------------------------------------------------------------------------*/
/* DATA STRUCTURES */
/*--------------------------------------------------------------------------*/

/* -- (none) -- */

/*--------------------------------------------------------------------------*/
/* B l o c k i n g D i s k  */
/*--------------------------------------------------------------------------*/

class Request
{
public:
    Thread *thread;
    Request *next;
};

class BlockingDisk : public SimpleDisk
{
private:
    DISK_ID disk_id;
    void issue_operation(DISK_OPERATION _op, unsigned long _block_no);

    void enqueue();
    void dequeue();

public:
    Request *head;
    Request *tail;
    bool disk_status;

    BlockingDisk(DISK_ID _disk_id, unsigned int _size);
    /* Creates a BlockingDisk device with the given size connected to the
       MASTER or SLAVE slot of the primary ATA controller.
       NOTE: We are passing the _size argument out of laziness.
       In a real system, we would infer this information from the
       disk controller. */

    /* DISK OPERATIONS */

    virtual void read(unsigned long _block_no, unsigned char *_buf);
    /* Reads 512 Bytes from the given block of the disk and copies them
       to the given buffer. No error check! */

    virtual void write(unsigned long _block_no, unsigned char *_buf);
    /* Writes 512 Bytes from the buffer to the given block on the disk. */

    virtual Thread *request_head();

    virtual bool is_ready();
};

#endif
