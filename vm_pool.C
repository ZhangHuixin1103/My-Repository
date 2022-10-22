/*
 File: vm_pool.C

 Author: Himanshu Gupta
 Date  : October 8 2018

 */

/*--------------------------------------------------------------------------*/
/* DEFINES */
/*--------------------------------------------------------------------------*/

/* -- (none) -- */

/*--------------------------------------------------------------------------*/
/* INCLUDES */
/*--------------------------------------------------------------------------*/

#include "page_table.H"
#include "vm_pool.H"
#include "console.H"
#include "utils.H"
#include "assert.H"
#include "simple_keyboard.H"

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

/* -- (none) -- */

/*--------------------------------------------------------------------------*/
/* METHODS FOR CLASS   V M P o o l */
/*--------------------------------------------------------------------------*/

VMPool::VMPool(unsigned long _base_address,
               unsigned long _size,
               ContFramePool *_frame_pool,
               PageTable *_page_table)
{
    base_address = _base_address;
    size = _size;
    frame_pool = _frame_pool;
    page_table = _page_table;

    region_num = 0;

    allocate_reg = (struct allocate_reg_ *)(base_address);

    page_table->register_pool(this);

    Console::puts("Constructed VMPool object.\n");
}

unsigned long VMPool::allocate(unsigned long _size)
{
    unsigned long address;
    unsigned b = _size % (Machine::PAGE_SIZE);
    unsigned long frames = _size / (Machine::PAGE_SIZE);

    if (b > 0)
        frames++;

    if (region_num == 0)
    {
        address = base_address;
        allocate_reg[region_num].base_address = address + Machine::PAGE_SIZE;
        allocate_reg[region_num].size = frames * (Machine::PAGE_SIZE);
        region_num++;
        return address + Machine::PAGE_SIZE;
    }
    else
    {
        address = allocate_reg[region_num - 1].base_address + allocate_reg[region_num - 1].size;
    }

    allocate_reg[region_num].base_address = address;
    allocate_reg[region_num].size = frames * (Machine::PAGE_SIZE);

    region_num++;

    Console::puts("Allocated region of memory.\n");

    return address;
}

void VMPool::release(unsigned long _start_address)
{
    int current_reg_num = -1;

    for (int i = 0; i < MAX_REGIONS; i++)
    {
        if (allocate_reg[i].base_address == _start_address)
        {
            current_reg_num = i;
            break;
        }
    }

    unsigned int allocate_pages = ((allocate_reg[current_reg_num].size) / (Machine::PAGE_SIZE));

    for (int i = 0; i < allocate_pages; i++)
    {
        page_table->free_page(_start_address);
        _start_address += Machine::PAGE_SIZE;
    }

    for (int i = current_reg_num; i < region_num - 1; i++)
    {
        allocate_reg[i] = allocate_reg[i + 1];
    }

    region_num--;

    page_table->load();

    Console::puts("Released region of memory.\n");
}

bool VMPool::is_legitimate(unsigned long _address)
{
    int size_l = base_address + size;
    int base_addr = base_address;
    if ((_address < size_l) && (_address >= base_addr))
        return true;
    return false;
}
